// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(phase)]

#[phase(plugin)]
extern crate green;
extern crate native;

use std::io::{TcpListener, Listener, Acceptor, EndOfFile, TcpStream};
use std::sync::{atomic, Arc};
use std::task::TaskBuilder;
use native::NativeTaskBuilder;

static N: uint = 8;
static M: uint = 100;

green_start!(main)

fn main() {
    test();

    let (tx, rx) = channel();
    TaskBuilder::new().native().spawn(proc() {
        tx.send(test());
    });
    rx.recv();
}

fn test() {
    let mut l = TcpListener::bind("127.0.0.1", 0).unwrap();
    let addr = l.socket_name().unwrap();
    let mut a = l.listen().unwrap();
    let cnt = Arc::new(atomic::AtomicUint::new(0));

    let (tx, rx) = channel();
    for _ in range(0, N) {
        let a = a.clone();
        let cnt = cnt.clone();
        let tx = tx.clone();
        spawn(proc() {
            let mut a = a;
            let mut mycnt = 0u;
            loop {
                match a.accept() {
                    Ok(..) => {
                        mycnt += 1;
                        if cnt.fetch_add(1, atomic::SeqCst) == N * M - 1 {
                            break
                        }
                    }
                    Err(ref e) if e.kind == EndOfFile => break,
                    Err(e) => fail!("{}", e),
                }
            }
            assert!(mycnt > 0);
            tx.send(());
        });
    }

    for _ in range(0, N) {
        let tx = tx.clone();
        spawn(proc() {
            for _ in range(0, M) {
                let _s = TcpStream::connect(addr.ip.to_string().as_slice(),
                                            addr.port).unwrap();
            }
            tx.send(());
        });
    }

    // wait for senders
    assert_eq!(rx.iter().take(N).count(), N);

    // wait for one acceptor to die
    let _ = rx.recv();

    // Notify other receivers should die
    a.close_accept().unwrap();

    // wait for receivers
    assert_eq!(rx.iter().take(N - 1).count(), N - 1);

    // Everything should have been accepted.
    assert_eq!(cnt.load(atomic::SeqCst), N * M);
}

