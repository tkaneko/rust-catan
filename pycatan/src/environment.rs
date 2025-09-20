use ndarray::Array1;
use pyo3::prelude::*;
 use pyo3::IntoPyObjectExt;
use numpy::convert::IntoPyArray;
use std::thread;
use std::sync::mpsc::{channel, Sender, Receiver};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use catan::game::Game;
use catan::state::State;
use catan::player::Randomy;
use catan::board::setup::random_default_setup_existing_state;
use catan::board::layout;
use super::{PythonState, PyCatanObservation, PyObservationFormat, PythonPlayer};

use std::sync::Mutex;


fn to_py_tuple(py: Python, hidden_state: bool, observation: Option<(u8, PyCatanObservation)>) -> PyObject {
    let elements: Vec<PyObject> = if let Some((id, observation)) = observation {
        if hidden_state {
            vec![
                // u8 -> PyInt -> PyObject
                id.into_py_any(py).unwrap(),

                // ndarray -> PyArray -> PyObject
                observation.board.into_pyarray(py).into(),
                observation.flat.into_pyarray(py).into(),
                observation.hidden.unwrap().into_pyarray(py).into(),
                observation.actions.into_pyarray(py).into(),
                false.into_py_any(py).unwrap(),
            ]
        } else {
            vec![
                id.into_py_any(py).unwrap(),
                observation.board.into_pyarray(py).into(),
                observation.flat.into_pyarray(py).into(),
                observation.actions.into_pyarray(py).into(),
                false.into_py_any(py).unwrap(),
            ]
        }
    } else {
        if hidden_state {
            vec![
                0i32.into_py_any(py).unwrap(),
                py.None(),
                py.None(),
                py.None(),
                py.None(),
                true.into_py_any(py).unwrap(),
            ]
        } else {
            vec![
                0i32.into_py_any(py).unwrap(),
                py.None(),
                py.None(),
                py.None(),
                true.into_py_any(py).unwrap(),
            ]
        }
    };

    // 型が統一されたVec<PyObject>からタプルを作成し、PyObjectに変換して返す
    elements.into_pyobject(py).unwrap().unbind().into_any()
}

#[pyclass]
pub struct SingleEnvironment {
    action_sender: Sender<u16>,
    observation_receiver: Mutex<Receiver<Option<(u8, PyCatanObservation)>>>,
    result_receiver: Mutex<Receiver<(u8,bool)>>,
    game_thread: thread::JoinHandle<()>,
    include_hidden: bool,
}

#[pymethods]
impl SingleEnvironment {

    #[staticmethod]
    #[pyo3(signature = (format, opponents=2))]
    fn new(format: &PyObservationFormat, opponents: usize) -> SingleEnvironment {
        let format = *format;
        let (action_sender, action_receiver) = channel();
        let (observation_sender, observation_receiver) = channel();
        let (result_sender, result_receiver) = channel();
        let game_thread = thread::spawn(move || {
            let mut game = Game::new();
            for _ in 0..opponents {
                game.add_player(Box::new(Randomy::new_player()));
            };
            game.add_player(Box::new(PythonPlayer::new(0, format, action_receiver, observation_sender, result_sender)));
            loop {
                game.setup_and_play();
            }
        });
        SingleEnvironment {
            action_sender,
            observation_receiver: Mutex::new(observation_receiver),
            result_receiver: Mutex::new(result_receiver),
            game_thread,
            include_hidden: format.include_hidden,
        }
    }

    fn start(&mut self, py: Python) -> PyResult<PyObject> {
        Ok(to_py_tuple(py, self.include_hidden, self.observation_receiver.lock().unwrap().recv().expect("Failed to read start observation")))
    }

    fn play(&mut self, py: Python, action: u16) -> PyResult<PyObject> {
        self.action_sender.send(action).expect("Failed to send action");
        self.game_thread.thread().unpark();
        Ok(to_py_tuple(py, self.include_hidden, self.observation_receiver.lock().unwrap().recv().expect("Failed to read play observation")))
    }

    fn result(&mut self, _py: Python) -> PyResult<(u8,bool)> {
        Ok(self.result_receiver.lock().unwrap().recv().expect("Failed to read results"))
    }
}


#[pyclass]
pub struct MultiEnvironment {
    players: usize,
    action_senders: Vec<Sender<u16>>,
    observation_receiver: Mutex<Receiver<Option<(u8, PyCatanObservation)>>>,
    result_receivers: Vec<Mutex<Receiver<(u8,bool)>>>,
    game_thread: thread::JoinHandle<()>,
    include_hidden: bool,
}

#[pymethods]
impl MultiEnvironment {

    #[staticmethod]
    #[pyo3(signature = (format, players=3))]
    fn new(format: &PyObservationFormat, players: usize) -> MultiEnvironment {
        let format = *format;
        let mut action_senders = Vec::new();
        let mut action_receivers = Vec::new();
        let mut result_senders = Vec::new();
        let mut result_receivers = Vec::new();
        for _ in 0..players {
            let (action_sender, action_receiver) = channel();
            let (result_sender, result_receiver) = channel();
            action_senders.push(action_sender);
            action_receivers.push(action_receiver);
            result_senders.push(result_sender);
            result_receivers.push(result_receiver);
        }
        let (observation_sender, observation_receiver) = channel();
        let game_thread = thread::spawn(move || {
            let mut game = Game::new();
            for (id, (action_receiver, result_sender)) in action_receivers.into_iter().zip(result_senders.into_iter()).enumerate() {
                game.add_player(Box::new(
                    PythonPlayer::new(id as u8, format, action_receiver, observation_sender.clone(), result_sender))
                );
            };
            let mut rng = SmallRng::from_rng(&mut rand::rng());
            loop {
                let mut state = PythonState::new(&layout::DEFAULT, players as u8, format);
                random_default_setup_existing_state::<PythonState, SmallRng>(&mut rng, &mut state);
                let mut players_order: Vec<usize> = (0..players).collect();
                players_order.shuffle(&mut rng);
                let mut state: State = Box::new(state);
                game.play(&mut rng, &mut state, players_order);
            }
        });
        MultiEnvironment {
            players,
            action_senders,
            observation_receiver: Mutex::new(observation_receiver),
            result_receivers: result_receivers.into_iter().map(Mutex::new).collect(),
            game_thread,
            include_hidden: format.include_hidden,
        }
    }

    fn start(&mut self, py: Python) -> PyResult<PyObject> {
        Ok(to_py_tuple(py, self.include_hidden, self.observation_receiver.lock().unwrap().recv().expect("Failed to read start observation")))
    }

    fn play(&mut self, py: Python, player: u8, action: u16) -> PyResult<PyObject> {
        self.action_senders[player as usize].send(action).expect("Failed to send action");
        self.game_thread.thread().unpark();
        Ok(to_py_tuple(py, self.include_hidden, self.observation_receiver.lock().unwrap().recv().expect("Failed to read play observation")))
    }

    fn result(&mut self, py: Python) -> PyResult<(PyObject, u8)> {
        let mut winner = 0;
        let mut vps = Array1::<u8>::zeros(self.players);
        for player in 0..self.players {
            let result = self.result_receivers[player].lock().unwrap().recv().expect("Failed to read results");
            vps[player] = result.0;
            if result.1 {
                winner = player;
            }
        }
        Ok((vps.into_pyarray(py).into_py_any(py).unwrap(), winner as u8))
    }
}
