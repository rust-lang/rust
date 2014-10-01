// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-linux see joyent/libuv#1189
// ignore-android needs extra network permissions
// exec-env:RUST_LOG=debug

#![feature(phase)]
#[phase(plugin, link)]
extern crate log;
extern crate libc;
extern crate debug;

use std::io::net::tcp::{TcpListener, TcpStream};
use std::io::{Acceptor, Listener};
use std::task::TaskBuilder;
use std::time::Duration;

fn main() {
    // This test has a chance to time out, try to not let it time out
    spawn(proc() {
        use std::io::timer;
        timer::sleep(Duration::milliseconds(30 * 1000));
        println!("timed out!");
        unsafe { libc::exit(1) }
    });

    let (tx, rx) = channel();
    spawn(proc() {
        let mut listener = TcpListener::bind("127.0.0.1", 0).unwrap();
        tx.send(listener.socket_name().unwrap());
        let mut acceptor = listener.listen();
        loop {
            let mut stream = match acceptor.accept() {
                Ok(stream) => stream,
                Err(error) => {
                    debug!("accept failed: {:?}", error);
                    continue;
                }
            };
            stream.read_byte();
            stream.write([2]);
        }
    });
    let addr = rx.recv();

    let (tx, rx) = channel();
    for _ in range(0u, 1000) {
        let tx = tx.clone();
        TaskBuilder::new().stack_size(64 * 1024).spawn(proc() {
            let host = addr.ip.to_string();
            let port = addr.port;
            match TcpStream::connect(host.as_slice(), port) {
                Ok(stream) => {
                    let mut stream = stream;
                    stream.write([1]);
                    let mut buf = [0];
                    stream.read(buf);
                },
                Err(e) => debug!("{:?}", e)
            }
            tx.send(());
        });
    }

    // Wait for all clients to exit, but don't wait for the server to exit. The
    // server just runs infinitely.
    drop(tx);
    for _ in range(0u, 1000) {
        rx.recv();
    }
    unsafe { libc::exit(0) }
}
