// xfail-pretty

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test performance of a task "spawn ladder", in which children task have many
// many ancestor taskgroups, but with only a few such groups alive at a time.
// Each child task has to enlist as a descendant in each of its ancestor
// groups, but that shouldn't have to happen for already-dead groups.
//
// The filename is a song reference; google it in quotes.

use std::cell::Cell;
use std::comm;
use std::os;
use std::task;
use std::uint;

fn child_generation(gens_left: uint, c: comm::Chan<()>) {
    // This used to be O(n^2) in the number of generations that ever existed.
    // With this code, only as many generations are alive at a time as tasks
    // alive at a time,
    let c = Cell::new(c);
    do task::spawn_supervised {
        let c = c.take();
        if gens_left & 1 == 1 {
            task::deschedule(); // shake things up a bit
        }
        if gens_left > 0 {
            child_generation(gens_left - 1, c); // recurse
        } else {
            c.send(())
        }
    }
}

fn main() {
    let args = os::args();
    let args = if os::getenv("RUST_BENCH").is_some() {
        ~[~"", ~"100000"]
    } else if args.len() <= 1 {
        ~[~"", ~"100"]
    } else {
        args.clone()
    };

    let (p,c) = comm::stream();
    child_generation(uint::from_str(args[1]).unwrap(), c);
    if p.try_recv().is_none() {
        fail!("it happened when we slumbered");
    }
}
