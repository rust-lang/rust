// This test creates a bunch of tasks that simultaneously send to each
// other in a ring. The messages should all be basically
// independent.
// This is like msgsend-ring-pipes but adapted to use ARCs.

// This also serves as a pipes test, because ARCs are implemented with pipes.

// xfail-pretty

import future::future;

use std;
import std::time;
import std::arc;

// A poor man's pipe.
type pipe = arc::mutex_arc<~[uint]>;

fn send(p: &pipe, msg: uint) {
    do p.access_cond |state, cond| {
        vec::push(*state, msg);
        cond.signal();
    }
}
fn recv(p: &pipe) -> uint {
    do p.access_cond |state, cond| {
        while vec::is_empty(*state) {
            cond.wait();
        }
        vec::pop(*state)
    }
}

fn init() -> (pipe,pipe) {
    let m = arc::mutex_arc(~[]);
    ((&m).clone(), m)
}


fn thread_ring(i: uint,
               count: uint,
               +num_chan: pipe,
               +num_port: pipe) {
    let mut num_chan <- some(num_chan);
    let mut num_port <- some(num_port);
    // Send/Receive lots of messages.
    for uint::range(0u, count) |j| {
        //error!{"task %?, iter %?", i, j};
        let mut num_chan2 = option::swap_unwrap(&mut num_chan);
        let mut num_port2 = option::swap_unwrap(&mut num_port);
        send(&num_chan2, i * j);
        num_chan = some(num_chan2);
        let _n = recv(&num_port2);
        //log(error, _n);
        num_port = some(num_port2);
    };
}

fn main(args: ~[~str]) {
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"100", ~"10000"]
    } else if args.len() <= 1u {
        ~[~"", ~"10", ~"100"]
    } else {
        copy args
    }; 

    let num_tasks = option::get(uint::from_str(args[1]));
    let msg_per_task = option::get(uint::from_str(args[2]));

    let (num_chan, num_port) = init();
    let mut num_chan = some(num_chan);

    let start = time::precise_time_s();

    // create the ring
    let mut futures = ~[];

    for uint::range(1u, num_tasks) |i| {
        //error!{"spawning %?", i};
        let (new_chan, num_port) = init();
        let num_chan2 = ~mut none;
        *num_chan2 <-> num_chan;
        let num_port = ~mut some(num_port);
        futures += ~[future::spawn(|move num_chan2, move num_port| {
            let mut num_chan = none;
            num_chan <-> *num_chan2;
            let mut num_port1 = none;
            num_port1 <-> *num_port;
            thread_ring(i, msg_per_task,
                        option::unwrap(num_chan),
                        option::unwrap(num_port1))
        })];
        num_chan = some(new_chan);
    };

    // do our iteration
    thread_ring(0u, msg_per_task, option::unwrap(num_chan), num_port);

    // synchronize
    for futures.each |f| { future::get(&f) };

    let stop = time::precise_time_s();

    // all done, report stats.
    let num_msgs = num_tasks * msg_per_task;
    let elapsed = (stop - start);
    let rate = (num_msgs as float) / elapsed;

    io::println(fmt!{"Sent %? messages in %? seconds",
                     num_msgs, elapsed});
    io::println(fmt!{"  %? messages / second", rate});
    io::println(fmt!{"  %? μs / message", 1000000. / rate});
}
