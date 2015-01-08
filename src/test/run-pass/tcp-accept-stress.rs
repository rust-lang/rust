// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-macos osx really doesn't like cycling through large numbers of
//              sockets as calls to connect() will start returning EADDRNOTAVAIL
//              quite quickly and it takes a few seconds for the sockets to get
//              recycled.

use std::io::{TcpListener, Listener, Acceptor, EndOfFile, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUint, Ordering};
use std::sync::mpsc::channel;
use std::thread::Thread;

static N: uint = 8;
static M: uint = 20;

fn main() {
    test();
}

fn test() {
    let mut l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.socket_name().unwrap();
    let mut a = l.listen().unwrap();
    let cnt = Arc::new(AtomicUint::new(0));

    let (srv_tx, srv_rx) = channel();
    let (cli_tx, cli_rx) = channel();
    let _t = range(0, N).map(|_| {
        let a = a.clone();
        let cnt = cnt.clone();
        let srv_tx = srv_tx.clone();
        Thread::scoped(move|| {
            let mut a = a;
            loop {
                match a.accept() {
                    Ok(..) => {
                        if cnt.fetch_add(1, Ordering::SeqCst) == N * M - 1 {
                            break
                        }
                    }
                    Err(ref e) if e.kind == EndOfFile => break,
                    Err(e) => panic!("{}", e),
                }
            }
            srv_tx.send(());
        })
    }).collect::<Vec<_>>();

    let _t = range(0, N).map(|_| {
        let cli_tx = cli_tx.clone();
        Thread::scoped(move|| {
            for _ in range(0, M) {
                let _s = TcpStream::connect(addr).unwrap();
            }
            cli_tx.send(());
        })
    }).collect::<Vec<_>>();
    drop((cli_tx, srv_tx));

    // wait for senders
    if cli_rx.iter().take(N).count() != N {
        a.close_accept().unwrap();
        panic!("clients panicked");
    }

    // wait for one acceptor to die
    let _ = srv_rx.recv();

    // Notify other receivers should die
    a.close_accept().unwrap();

    // wait for receivers
    assert_eq!(srv_rx.iter().take(N - 1).count(), N - 1);

    // Everything should have been accepted.
    assert_eq!(cnt.load(Ordering::SeqCst), N * M);
}
