// This test creates a bunch of tasks that simultaneously send to each
// other in a ring. The messages should all be basically
// independent. It's designed to hammer the global kernel lock, so
// that things will look really good once we get that lock out of the
// message path.

use comm::*;
use future::future;

extern mod std;
use std::time;

fn thread_ring(i: uint,
               count: uint,
               num_chan: comm::Chan<uint>,
               num_port: comm::Port<uint>) {
    // Send/Receive lots of messages.
    for uint::range(0u, count) |j| {
        num_chan.send(i * j);
        num_port.recv();
    };
}

fn main(++args: ~[~str]) {
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"100", ~"10000"]
    } else if args.len() <= 1u {
        ~[~"", ~"100", ~"1000"]
    } else {
        args
    };        

    let num_tasks = uint::from_str(args[1]).get();
    let msg_per_task = uint::from_str(args[2]).get();

    let num_port = Port();
    let mut num_chan = Chan(num_port);

    let start = time::precise_time_s();

    // create the ring
    let mut futures = ~[];

    for uint::range(1u, num_tasks) |i| {
        let get_chan = Port();
        let get_chan_chan = Chan(get_chan);

        let new_future = do future::spawn
            |copy num_chan, move get_chan_chan| {
            let p = Port();
            get_chan_chan.send(Chan(p));
            thread_ring(i, msg_per_task, num_chan,  p)
        };
        vec::push(futures, new_future);
        
        num_chan = get_chan.recv();
    };

    // do our iteration
    thread_ring(0u, msg_per_task, num_chan, num_port);

    // synchronize
    for futures.each |f| { f.get() };

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
