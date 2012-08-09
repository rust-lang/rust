// A port of the simplistic benchmark from
//
//    http://github.com/PaulKeeble/ScalaVErlangAgents
//
// I *think* it's the same, more or less.

use std;
import io::writer;
import io::writer_util;

enum request {
    get_count,
    bytes(uint),
    stop
}

fn server(requests: comm::port<request>, responses: comm::chan<uint>) {
    let mut count = 0u;
    let mut done = false;
    while !done {
        match comm::recv(requests) {
          get_count => { comm::send(responses, copy count); }
          bytes(b) => { count += b; }
          stop => { done = true; }
        }
    }
    comm::send(responses, count);
}

fn run(args: ~[~str]) {
    let (from_child, to_child) = do task::spawn_conversation |po, ch| {
        server(po, ch);
    };
    let size = option::get(uint::from_str(args[1]));
    let workers = option::get(uint::from_str(args[2]));
    let start = std::time::precise_time_s();
    let mut worker_results = ~[];
    for uint::range(0u, workers) |_i| {
        do task::task().future_result(|+r| {
            vec::push(worker_results, r);
        }).spawn {
            for uint::range(0u, size / workers) |_i| {
                comm::send(to_child, bytes(100u));
            }
        };
    }
    vec::iter(worker_results, |r| { future::get(r); } );
    comm::send(to_child, stop);
    let result = comm::recv(from_child);
    let end = std::time::precise_time_s();
    let elapsed = end - start;
    io::stdout().write_str(fmt!{"Count is %?\n", result});
    io::stdout().write_str(fmt!{"Test took %? seconds\n", elapsed});
    let thruput = ((size / workers * workers) as float) / (elapsed as float);
    io::stdout().write_str(fmt!{"Throughput=%f per sec\n", thruput});
}

fn main(args: ~[~str]) {
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"1000000", ~"10000"]
    } else if args.len() <= 1u {
        ~[~"", ~"10000", ~"4"]
    } else {
        args
    };

    debug!{"%?", args};
    run(args);
}

