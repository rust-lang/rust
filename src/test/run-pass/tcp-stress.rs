// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android needs extra network permissions
// ignore-openbsd system ulimit (Too many open files)
// ignore-bitrig system ulimit (Too many open files)

use std::io::prelude::*;
use std::net::{TcpListener, TcpStream};
use std::process;
use std::sync::mpsc::channel;
use std::thread::{self, Builder};

fn main() {
    // This test has a chance to time out, try to not let it time out
    thread::spawn(move|| -> () {
        thread::sleep_ms(30 * 1000);
        process::exit(1);
    });

    let mut listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || -> () {
        loop {
            let mut stream = match listener.accept() {
                Ok(stream) => stream.0,
                Err(error) => continue,
            };
            stream.read(&mut [0]);
            stream.write(&[2]);
        }
    });

    let (tx, rx) = channel();
    for _ in 0..1000 {
        let tx = tx.clone();
        Builder::new().stack_size(64 * 1024).spawn(move|| {
            match TcpStream::connect(addr) {
                Ok(mut stream) => {
                    stream.write(&[1]);
                    stream.read(&mut [0]);
                },
                Err(..) => {}
            }
            tx.send(()).unwrap();
        });
    }

    // Wait for all clients to exit, but don't wait for the server to exit. The
    // server just runs infinitely.
    drop(tx);
    for _ in 0..1000 {
        rx.recv().unwrap();
    }
    process::exit(0);
}
