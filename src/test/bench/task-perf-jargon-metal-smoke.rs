// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test performance of a task "spawn ladder", in which children task have
// many ancestor taskgroups, but with only a few such groups alive at a time.
// Each child task has to enlist as a descendant in each of its ancestor
// groups, but that shouldn't have to happen for already-dead groups.
//
// The filename is a song reference; google it in quotes.

// ignore-pretty very bad with line comments

use std::comm;
use std::os;
use std::task;
use std::uint;

fn child_generation(gens_left: uint, tx: comm::Sender<()>) {
    // This used to be O(n^2) in the number of generations that ever existed.
    // With this code, only as many generations are alive at a time as tasks
    // alive at a time,
    spawn(proc() {
        if gens_left & 1 == 1 {
            task::deschedule(); // shake things up a bit
        }
        if gens_left > 0 {
            child_generation(gens_left - 1, tx); // recurse
        } else {
            tx.send(())
        }
    });
}

fn main() {
    let args = os::args();
    let args = if os::getenv("RUST_BENCH").is_some() {
        vec!("".to_owned(), "100000".to_owned())
    } else if args.len() <= 1 {
        vec!("".to_owned(), "100".to_owned())
    } else {
        args.clone().move_iter().collect()
    };

    let (tx, rx) = channel();
    child_generation(from_str::<uint>(args.get(1).as_slice()).unwrap(), tx);
    if rx.recv_opt().is_err() {
        fail!("it happened when we slumbered");
    }
}
