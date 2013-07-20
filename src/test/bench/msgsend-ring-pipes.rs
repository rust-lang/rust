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
// independent. It's designed to hammer the global kernel lock, so
// that things will look really good once we get that lock out of the
// message path.

// This version uses automatically compiled channel contracts.

extern mod extra;

use extra::future;
use extra::time;
use std::cell::Cell;
use std::io;
use std::os;
use std::pipes::recv;
use std::uint;
use std::util;

proto! ring (
    num:send {
        num(uint) -> num
    }
)

fn thread_ring(i: uint,
               count: uint,
               num_chan: ring::client::num,
               num_port: ring::server::num) {
    let mut num_chan = Some(num_chan);
    let mut num_port = Some(num_port);
    // Send/Receive lots of messages.
    for uint::range(0, count) |j| {
        //error!("task %?, iter %?", i, j);
        let num_chan2 = util::replace(&mut num_chan, None);
        let num_port2 = util::replace(&mut num_port, None);
        num_chan = Some(ring::client::num(num_chan2.unwrap(), i * j));
        let port = num_port2.unwrap();
        match recv(port) {
          ring::num(_n, p) => {
            //log(error, _n);
            num_port = Some(p);
          }
        }
    };
}

fn main() {
    let args = os::args();
    let args = if os::getenv("RUST_BENCH").is_some() {
        ~[~"", ~"100", ~"10000"]
    } else if args.len() <= 1u {
        ~[~"", ~"100", ~"1000"]
    } else {
        args.clone()
    };

    let num_tasks = uint::from_str(args[1]).get();
    let msg_per_task = uint::from_str(args[2]).get();

    let (num_port, num_chan) = ring::init();
    let num_chan = Cell::new(num_chan);

    let start = time::precise_time_s();

    // create the ring
    let mut futures = ~[];

    for uint::range(1u, num_tasks) |i| {
        //error!("spawning %?", i);
        let (num_port, new_chan) = ring::init();
        let num_chan2 = Cell::new(num_chan.take());
        let num_port = Cell::new(num_port);
        let new_future = do future::spawn || {
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
    for futures.mut_iter().advance |f| {
        let _ = f.get();
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
