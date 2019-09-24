// run-pass
#![allow(unused_must_use)]
#![allow(deprecated)]
// ignore-emscripten no threads support
// ignore-sgx no thread sleep support

use std::sync::mpsc::{TryRecvError, channel};
use std::thread;

pub fn main() {
    let (tx, rx) = channel();
    let t = thread::spawn(move||{
        thread::sleep_ms(10);
        tx.send(()).unwrap();
    });
    loop {
        match rx.try_recv() {
            Ok(()) => break,
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => unreachable!()
        }
    }
    t.join();
}
