// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
