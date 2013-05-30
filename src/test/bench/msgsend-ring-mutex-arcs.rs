// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test creates a bunch of tasks that simultaneously send to each
// other in a ring. The messages should all be basically
// independent.
// This is like msgsend-ring-pipes but adapted to use ARCs.

// This also serves as a pipes test, because ARCs are implemented with pipes.

extern mod extra;

use extra::arc;
use extra::future;
use extra::time;
use std::cell::Cell;
use std::io;
use std::os;
use std::uint;
use std::vec;

// A poor man's pipe.
type pipe = arc::MutexARC<~[uint]>;

fn send(p: &pipe, msg: uint) {
    unsafe {
        do p.access_cond |state, cond| {
            state.push(msg);
            cond.signal();
        }
    }
}
fn recv(p: &pipe) -> uint {
    unsafe {
        do p.access_cond |state, cond| {
            while vec::is_empty(*state) {
                cond.wait();
            }
            state.pop()
        }
    }
}

fn init() -> (pipe,pipe) {
    let m = arc::MutexARC(~[]);
    ((&m).clone(), m)
}


fn thread_ring(i: uint, count: uint, num_chan: pipe, num_port: pipe) {
    let mut num_chan = Some(num_chan);
    let mut num_port = Some(num_port);
    // Send/Receive lots of messages.
    for uint::range(0u, count) |j| {
        //error!("task %?, iter %?", i, j);
        let mut num_chan2 = num_chan.swap_unwrap();
        let mut num_port2 = num_port.swap_unwrap();
        send(&num_chan2, i * j);
        num_chan = Some(num_chan2);
        let _n = recv(&num_port2);
        //log(error, _n);
        num_port = Some(num_port2);
    };
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"100", ~"10000"]
    } else if args.len() <= 1u {
        ~[~"", ~"10", ~"100"]
    } else {
        copy args
    };

    let num_tasks = uint::from_str(args[1]).get();
    let msg_per_task = uint::from_str(args[2]).get();

    let (num_chan, num_port) = init();
    let mut num_chan = Cell(num_chan);

    let start = time::precise_time_s();

    // create the ring
    let mut futures = ~[];

    for uint::range(1u, num_tasks) |i| {
        //error!("spawning %?", i);
        let (new_chan, num_port) = init();
        let num_chan2 = Cell(num_chan.take());
        let num_port = Cell(num_port);
        let new_future = do future::spawn() {
            let num_chan = num_chan2.take();
            let num_port1 = num_port.take();
            thread_ring(i, msg_per_task, num_chan, num_port1)
        };
        futures.push(new_future);
        num_chan.put_back(new_chan);
    };

    // do our iteration
    thread_ring(0, msg_per_task, num_chan.take(), num_port);

    // synchronize
    for futures.each_mut |f| {
        f.get()
    }

    let stop = time::precise_time_s();

    // all done, report stats.
    let num_msgs = num_tasks * msg_per_task;
    let elapsed = (stop - start);
    let rate = (num_msgs as float) / elapsed;

    io::println(fmt!("Sent %? messages in %? seconds",
                     num_msgs, elapsed));
    io::println(fmt!("  %? messages / second", rate));
    io::println(fmt!("  %? μs / message", 1000000. / rate));
}
