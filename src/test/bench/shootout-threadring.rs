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
extern mod std;

const n_threads: int = 503;

fn start(+token: int) {
    use iter::*;

    let p = comm::Port();
    let mut ch = comm::Chan(&p);
    for int::range(2, n_threads + 1) |i| {
        let id = n_threads + 2 - i;
        let to_child = do task::spawn_listener::<int> |p, copy ch| {
            roundtrip(id, p, ch)
        };
        ch = to_child;
    }
    comm::send(ch, token);
    roundtrip(1, p, ch);
}

fn roundtrip(id: int, p: comm::Port<int>, ch: comm::Chan<int>) {
    while (true) {
        match comm::recv(p) {
          1 => {
            io::println(fmt!("%d\n", id));
            return;
          }
          token => {
            debug!("%d %d", id, token);
            comm::send(ch, token - 1);
            if token <= n_threads {
                return;
            }
          }
        }
    }
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"2000000"]
    } else if args.len() <= 1u {
        ~[~"", ~"1000"]
    } else {
        args
    };

    let token = int::from_str(args[1]).get();

    start(token);
}
