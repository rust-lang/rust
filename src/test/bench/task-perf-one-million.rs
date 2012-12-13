// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test for concurrent tasks

enum msg {
    ready(oldcomm::Chan<msg>),
    start,
    done(int),
}

fn calc(children: uint, parent_ch: oldcomm::Chan<msg>) {
    let port = oldcomm::Port();
    let chan = oldcomm::Chan(&port);
    let mut child_chs = ~[];
    let mut sum = 0;

    for iter::repeat (children) {
        do task::spawn {
            calc(0u, chan);
        };
    }

    for iter::repeat (children) {
        match oldcomm::recv(port) {
          ready(child_ch) => {
            child_chs.push(child_ch);
          }
          _ => fail ~"task-perf-one-million failed (port not ready)"
        }
    }

    oldcomm::send(parent_ch, ready(chan));

    match oldcomm::recv(port) {
        start => {
            for vec::each(child_chs) |child_ch| {
                oldcomm::send(*child_ch, start);
            }
        }
        _ => fail ~"task-perf-one-million failed (port not in start state)"
    }

    for iter::repeat (children) {
        match oldcomm::recv(port) {
          done(child_sum) => { sum += child_sum; }
          _ => fail ~"task-perf-one-million failed (port not done)"
        }
    }

    oldcomm::send(parent_ch, done(sum + 1));
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"100000"]
    } else if args.len() <= 1u {
        ~[~"", ~"100"]
    } else {
        args
    };

    let children = uint::from_str(args[1]).get();
    let port = oldcomm::Port();
    let chan = oldcomm::Chan(&port);
    do task::spawn {
        calc(children, chan);
    };
    match oldcomm::recv(port) {
      ready(chan) => {
        oldcomm::send(chan, start);
      }
      _ => fail ~"task-perf-one-million failed (port not ready)"
    }
    let sum = match oldcomm::recv(port) {
      done(sum) => { sum }
      _ => fail ~"task-perf-one-million failed (port not done)"
    };
    error!("How many tasks? %d tasks.", sum);
}
