// A port of the simplistic benchmark from
//
//    http://github.com/PaulKeeble/ScalaVErlangAgents
//
// I *think* it's the same, more or less.

// This version uses pipes with a shared send endpoint. It should have
// different scalability characteristics compared to the select
// version.

// xfail-pretty

extern mod std;
use io::Writer;
use io::WriterUtil;

use pipes::{Port, Chan, SharedChan};

macro_rules! move_out (
    { $x:expr } => { unsafe { let y <- *ptr::addr_of($x); y } }
)

enum request {
    get_count,
    bytes(uint),
    stop
}

fn server(requests: Port<request>, responses: pipes::Chan<uint>) {
    let mut count = 0u;
    let mut done = false;
    while !done {
        match requests.try_recv() {
          Some(get_count) => { responses.send(copy count); }
          Some(bytes(b)) => {
            //error!("server: received %? bytes", b);
            count += b;
          }
          None => { done = true; }
          _ => { }
        }
    }
    responses.send(count);
    //error!("server exiting");
}

fn run(args: &[~str]) {
    let (to_parent, from_child) = pipes::stream();
    let (to_child, from_parent) = pipes::stream();

    let to_child = SharedChan(to_child);

    let size = option::get(uint::from_str(args[1]));
    let workers = option::get(uint::from_str(args[2]));
    let num_bytes = 100;
    let start = std::time::precise_time_s();
    let mut worker_results = ~[];
    for uint::range(0u, workers) |i| {
        let to_child = to_child.clone();
        do task::task().future_result(|+r| {
            vec::push(worker_results, r);
        }).spawn {
            for uint::range(0u, size / workers) |_i| {
                //error!("worker %?: sending %? bytes", i, num_bytes);
                to_child.send(bytes(num_bytes));
            }
            //error!("worker %? exiting", i);
        };
    }
    do task::spawn {
        server(from_parent, to_parent);
    }

    vec::iter(worker_results, |r| { future::get(&r); } );
    //error!("sending stop message");
    to_child.send(stop);
    move_out!(to_child);
    let result = from_child.recv();
    let end = std::time::precise_time_s();
    let elapsed = end - start;
    io::stdout().write_str(fmt!("Count is %?\n", result));
    io::stdout().write_str(fmt!("Test took %? seconds\n", elapsed));
    let thruput = ((size / workers * workers) as float) / (elapsed as float);
    io::stdout().write_str(fmt!("Throughput=%f per sec\n", thruput));
    assert result == num_bytes * size;
}

fn main(args: ~[~str]) {
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"1000000", ~"10000"]
    } else if args.len() <= 1u {
        ~[~"", ~"10000", ~"4"]
    } else {
        copy args
    };        

    debug!("%?", args);
    run(args);
}
