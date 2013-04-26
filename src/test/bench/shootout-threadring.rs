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

// xfail-test FIXME #5985 OOM's on the mac bot

fn start(n_tasks: int, token: int) {
    let mut (p, ch1) = comm::stream();
    ch1.send(token);
    //  XXX could not get this to work with a range closure
    let mut i = 2;
    while i <= n_tasks {
        let (next_p, ch) = comm::stream();
        let imm_i = i;
        let imm_p = p;
        do task::spawn {
            roundtrip(imm_i, n_tasks, &imm_p, &ch);
        };
        p = next_p;
        i += 1;
    }
    let imm_p = p;
    let imm_ch = ch1;
    do task::spawn {
        roundtrip(1, n_tasks, &imm_p, &imm_ch);
    }
}

fn roundtrip(id: int, n_tasks: int, p: &comm::Port<int>, ch: &comm::Chan<int>) {
    while (true) {
        match p.recv() {
          1 => {
            io::println(fmt!("%d\n", id));
            return;
          }
          token => {
            debug!("thread: %d   got token: %d", id, token);
            ch.send(token - 1);
            if token <= n_tasks {
                return;
            }
          }
        }
    }
}

fn main() {
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"2000000", ~"503"]
    }
    else {
        os::args()
    };
    let token = if args.len() > 1u {
        int::from_str(args[1]).get()
    }
    else {
        1000
    };
    let n_tasks = if args.len() > 2u {
        int::from_str(args[2]).get()
    }
    else {
        503
    };
    start(n_tasks, token);

}
