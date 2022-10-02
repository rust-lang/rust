// run-pass
// ignore-emscripten no threads support
// needs-unwind

#![feature(internal_output_capture)]

use std::io;
use std::str;
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let data = Arc::new(Mutex::new(Vec::new()));
    let res = thread::Builder::new().spawn({
        let data = data.clone();
        move || {
            io::set_output_capture(Some(data));
            panic!("Hello, world!")
        }
    }).unwrap().join();
    assert!(res.is_err());

    let output = data.lock().unwrap();
    let output = str::from_utf8(&output).unwrap();
    assert!(output.contains("Hello, world!"));
}
