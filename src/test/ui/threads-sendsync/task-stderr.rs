// run-pass
// ignore-emscripten no threads support

#![feature(box_syntax, set_stdio)]

use std::io::prelude::*;
use std::io;
use std::str;
use std::sync::{Arc, Mutex};
use std::thread;

struct Sink(Arc<Mutex<Vec<u8>>>);
impl Write for Sink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        Write::write(&mut *self.0.lock().unwrap(), data)
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
impl io::LocalOutput for Sink {
    fn clone_box(&self) -> Box<dyn io::LocalOutput> {
        Box::new(Sink(self.0.clone()))
    }
}

fn main() {
    let data = Arc::new(Mutex::new(Vec::new()));
    let sink = Sink(data.clone());
    let res = thread::Builder::new().spawn(move|| -> () {
        io::set_panic(Some(Box::new(sink)));
        panic!("Hello, world!")
    }).unwrap().join();
    assert!(res.is_err());

    let output = data.lock().unwrap();
    let output = str::from_utf8(&output).unwrap();
    assert!(output.contains("Hello, world!"));
}
