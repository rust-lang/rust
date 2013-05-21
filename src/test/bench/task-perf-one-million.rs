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

use std::comm::*;

fn calc(children: uint, parent_wait_chan: &Chan<Chan<Chan<int>>>) {

    let wait_ports: ~[Port<Chan<Chan<int>>>] = do vec::from_fn(children) |_| {
        let (wait_port, wait_chan) = stream::<Chan<Chan<int>>>();
        do task::spawn {
            calc(children / 2, &wait_chan);
        }
        wait_port
    };

    let child_start_chans: ~[Chan<Chan<int>>] = vec::map_consume(wait_ports, |port| port.recv());

    let (start_port, start_chan) = stream::<Chan<int>>();
    parent_wait_chan.send(start_chan);
    let parent_result_chan: Chan<int> = start_port.recv();

    let child_sum_ports: ~[Port<int>] = do vec::map_consume(child_start_chans) |child_start_chan| {
        let (child_sum_port, child_sum_chan) = stream::<int>();
        child_start_chan.send(child_sum_chan);
        child_sum_port
    };

    let mut sum = 0;
    vec::consume(child_sum_ports, |_, sum_port| sum += sum_port.recv() );

    parent_result_chan.send(sum + 1);
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"30"]
    } else if args.len() <= 1u {
        ~[~"", ~"10"]
    } else {
        args
    };

    let children = uint::from_str(args[1]).get();
    let (wait_port, wait_chan) = stream();
    do task::spawn {
        calc(children, &wait_chan);
    };

    let start_chan = wait_port.recv();
    let (sum_port, sum_chan) = stream::<int>();
    start_chan.send(sum_chan);
    let sum = sum_port.recv();
    error!("How many tasks? %d tasks.", sum);
}
