use std::thread;
use std::sync::mpsc::{Receiver, channel};

pub fn foo<T:'static + Send + Clone>(x: T) -> Receiver<T> {
    let (tx, rx) = channel();
    thread::spawn(move|| {
        tx.send(x.clone());
    });
    rx
}
