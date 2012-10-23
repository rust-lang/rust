// A port of the simplistic benchmark from
//
//    http://github.com/PaulKeeble/ScalaVErlangAgents
//
// I *think* it's the same, more or less.

// xfail-pretty

#[legacy_modes];

extern mod std;
use io::Writer;
use io::WriterUtil;

use pipes::{Port, PortSet, Chan};

macro_rules! move_out (
    { $x:expr } => { unsafe { let y = move *ptr::addr_of(&($x)); move y } }
)

enum request {
    get_count,
    bytes(uint),
    stop
}

fn server(requests: PortSet<request>, responses: pipes::Chan<uint>) {
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
    let (to_child, from_parent_) = pipes::stream();
    let from_parent = PortSet();
    from_parent.add(move from_parent_);

    let size = uint::from_str(args[1]).get();
    let workers = uint::from_str(args[2]).get();
    let num_bytes = 100;
    let start = std::time::precise_time_s();
    let mut worker_results = ~[];
    for uint::range(0, workers) |_i| {
        let (to_child, from_parent_) = pipes::stream();
        from_parent.add(move from_parent_);
        do task::task().future_result(|+r| {
            worker_results.push(move r);
        }).spawn |move to_child| {
            for uint::range(0, size / workers) |_i| {
                //error!("worker %?: sending %? bytes", i, num_bytes);
                to_child.send(bytes(num_bytes));
            }
            //error!("worker %? exiting", i);
        };
    }
    do task::spawn |move from_parent, move to_parent| {
        server(from_parent, to_parent);
    }

    for vec::each(worker_results) |r| {
        future::get(r);
    }

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

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"1000000", ~"8"]
    } else if args.len() <= 1u {
        ~[~"", ~"10000", ~"4"]
    } else {
        copy args
    };        

    debug!("%?", args);
    run(args);
}
