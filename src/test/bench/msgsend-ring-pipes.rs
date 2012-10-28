// This test creates a bunch of tasks that simultaneously send to each
// other in a ring. The messages should all be basically
// independent. It's designed to hammer the global kernel lock, so
// that things will look really good once we get that lock out of the
// message path.

// This version uses automatically compiled channel contracts.

extern mod std;
use std::time;
use std::future;

use pipes::recv;

proto! ring (
    num:send {
        num(uint) -> num
    }
)

fn macros() {
    #macro[
        [#move_out[x],
         unsafe { let y = move *ptr::addr_of(&x); move y }]
    ];
}

fn thread_ring(i: uint,
               count: uint,
               +num_chan: ring::client::num,
               +num_port: ring::server::num) {
    let mut num_chan = move Some(move num_chan);
    let mut num_port = move Some(move num_port);
    // Send/Receive lots of messages.
    for uint::range(0, count) |j| {
        //error!("task %?, iter %?", i, j);
        let mut num_chan2 = None;
        let mut num_port2 = None;
        num_chan2 <-> num_chan;
        num_port2 <-> num_port;
        num_chan = Some(ring::client::num(option::unwrap(move num_chan2), i * j));
        let port = option::unwrap(move num_port2);
        match recv(move port) {
          ring::num(_n, p) => {
            //log(error, _n);
            num_port = Some(move_out!(p));
          }
        }
    };
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"100", ~"10000"]
    } else if args.len() <= 1u {
        ~[~"", ~"100", ~"1000"]
    } else {
        copy args
    }; 

    let num_tasks = uint::from_str(args[1]).get();
    let msg_per_task = uint::from_str(args[2]).get();

    let (num_chan, num_port) = ring::init();
    let mut num_chan = Some(move num_chan);

    let start = time::precise_time_s();

    // create the ring
    let mut futures = ~[];

    for uint::range(1u, num_tasks) |i| {
        //error!("spawning %?", i);
        let (new_chan, num_port) = ring::init();
        let num_chan2 = ~mut None;
        *num_chan2 <-> num_chan;
        let num_port = ~mut Some(move num_port);
        let new_future = do future::spawn
            |move num_chan2, move num_port| {
            let mut num_chan = None;
            num_chan <-> *num_chan2;
            let mut num_port1 = None;
            num_port1 <-> *num_port;
            thread_ring(i, msg_per_task,
                        option::unwrap(move num_chan),
                        option::unwrap(move num_port1))
        };
        futures.push(move new_future);
        num_chan = Some(move new_chan);
    };

    // do our iteration
    thread_ring(0, msg_per_task, option::unwrap(move num_chan), move num_port);

    // synchronize
    for futures.each |f| { future::get(f) };

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
