// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test creates a bunch of threads that simultaneously send to each
// other in a ring. The messages should all be basically
// independent.
// This is like msgsend-ring-pipes but adapted to use Arcs.

// This also serves as a pipes test, because Arcs are implemented with pipes.

// no-pretty-expanded FIXME #15189

#![feature(time2)]

use std::env;
use std::sync::{Arc, Mutex, Condvar};
use std::time::Instant;
use std::thread;

// A poor man's pipe.
type pipe = Arc<(Mutex<Vec<usize>>, Condvar)>;

fn send(p: &pipe, msg: usize) {
    let &(ref lock, ref cond) = &**p;
    let mut arr = lock.lock().unwrap();
    arr.push(msg);
    cond.notify_one();
}
fn recv(p: &pipe) -> usize {
    let &(ref lock, ref cond) = &**p;
    let mut arr = lock.lock().unwrap();
    while arr.is_empty() {
        arr = cond.wait(arr).unwrap();
    }
    arr.pop().unwrap()
}

fn init() -> (pipe,pipe) {
    let m = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
    ((&m).clone(), m)
}


fn thread_ring(i: usize, count: usize, num_chan: pipe, num_port: pipe) {
    let mut num_chan = Some(num_chan);
    let mut num_port = Some(num_port);
    // Send/Receive lots of messages.
    for j in 0..count {
        //println!("thread %?, iter %?", i, j);
        let num_chan2 = num_chan.take().unwrap();
        let num_port2 = num_port.take().unwrap();
        send(&num_chan2, i * j);
        num_chan = Some(num_chan2);
        let _n = recv(&num_port2);
        //log(error, _n);
        num_port = Some(num_port2);
    };
}

fn main() {
    let args = env::args();
    let args = if env::var_os("RUST_BENCH").is_some() {
        vec!("".to_string(), "100".to_string(), "10000".to_string())
    } else if args.len() <= 1 {
        vec!("".to_string(), "10".to_string(), "100".to_string())
    } else {
        args.collect()
    };

    let num_tasks = args[1].parse::<usize>().unwrap();
    let msg_per_task = args[2].parse::<usize>().unwrap();

    let (num_chan, num_port) = init();

    let mut p = Some((num_chan, num_port));
    let start = Instant::now();
    {
        let (mut num_chan, num_port) = p.take().unwrap();

        // create the ring
        let mut futures = Vec::new();

        for i in 1..num_tasks {
            //println!("spawning %?", i);
            let (new_chan, num_port) = init();
            let num_chan_2 = num_chan.clone();
            let new_future = thread::spawn(move|| {
                thread_ring(i, msg_per_task, num_chan_2, num_port)
            });
            futures.push(new_future);
            num_chan = new_chan;
        };

        // do our iteration
        thread_ring(0, msg_per_task, num_chan, num_port);

        // synchronize
        for f in futures {
            f.join().unwrap()
        }
    }
    let dur = start.elapsed();

    // all done, report stats.
    let num_msgs = num_tasks * msg_per_task;
    let rate = (num_msgs as f64) / (dur.as_secs() as f64);

    println!("Sent {} messages in {:?}", num_msgs, dur);
    println!("  {} messages / second", rate);
    println!("  {} μs / message", 1000000. / rate);
}
