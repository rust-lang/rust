// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Based on threadring.erlang by Jira Isa

use std::os;

fn start(n_tasks: int, token: int) {
    let (tx, mut rx) = channel();
    tx.send(token);
    //  XXX could not get this to work with a range closure
    let mut i = 2;
    while i <= n_tasks {
        let (tx, next_rx) = channel();
        let imm_i = i;
        let imm_rx = rx;
        spawn(proc() {
            roundtrip(imm_i, n_tasks, &imm_rx, &tx);
        });
        rx = next_rx;
        i += 1;
    }
    let imm_rx = rx;
    spawn(proc() {
        roundtrip(1, n_tasks, &imm_rx, &tx);
    });
}

fn roundtrip(id: int, n_tasks: int, p: &Receiver<int>, ch: &Sender<int>) {
    loop {
        match p.recv() {
          1 => {
            println!("{}\n", id);
            return;
          }
          token => {
            println!("thread: {}   got token: {}", id, token);
            ch.send(token - 1);
            if token <= n_tasks {
                return;
            }
          }
        }
    }
}

fn main() {
    use std::from_str::FromStr;

    let args = if os::getenv("RUST_BENCH").is_some() {
        ~[~"", ~"2000000", ~"503"]
    }
    else {
        os::args()
    };
    let token = if args.len() > 1u {
        FromStr::from_str(args[1]).unwrap()
    }
    else {
        1000
    };
    let n_tasks = if args.len() > 2u {
        FromStr::from_str(args[2]).unwrap()
    }
    else {
        503
    };
    start(n_tasks, token);

}
