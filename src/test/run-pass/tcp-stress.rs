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
// ignore-fast
// ignore-android needs extra network permissions
// exec-env:RUST_LOG=debug

use std::libc;
use std::io::net::ip::{Ipv4Addr, SocketAddr};
use std::io::net::tcp::{TcpListener, TcpStream};
use std::io::{Acceptor, Listener};

fn main() {
    // This test has a chance to time out, try to not let it time out
    spawn(proc() {
        use std::io::timer;
        timer::sleep(30 * 1000);
        println!("timed out!");
        unsafe { libc::exit(1) }
    });

    let addr = SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: 0 };
    let (p, c) = Chan::new();
    spawn(proc() {
        let mut listener = TcpListener::bind(addr).unwrap();
        c.send(listener.socket_name().unwrap());
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
    let addr = p.recv();

    let (p, c) = Chan::new();
    for _ in range(0, 1000) {
        let c = c.clone();
        spawn(proc() {
            match TcpStream::connect(addr) {
                Ok(stream) => {
                    let mut stream = stream;
                    stream.write([1]);
                    let mut buf = [0];
                    stream.read(buf);
                },
                Err(e) => debug!("{:?}", e)
            }
            c.send(());
        });
    }

    // Wait for all clients to exit, but don't wait for the server to exit. The
    // server just runs infinitely.
    drop(c);
    for _ in range(0, 1000) {
        p.recv();
    }
    unsafe { libc::exit(0) }
}
