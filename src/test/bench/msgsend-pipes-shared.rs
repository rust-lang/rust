// A port of the simplistic benchmark from
//
//    http://github.com/PaulKeeble/ScalaVErlangAgents
//
// I *think* it's the same, more or less.

// This version uses pipes with a shared send endpoint. It should have
// different scalability characteristics compared to the select
// version.

use std;
import io::writer;
import io::writer_util;

import arc::methods;
import pipes::{port, chan};

macro_rules! move {
    { $x:expr } => { unsafe { let y <- *ptr::addr_of($x); y } }
}

enum request {
    get_count,
    bytes(uint),
    stop
}

fn server(requests: port<request>, responses: pipes::chan<uint>) {
    let mut count = 0u;
    let mut done = false;
    while !done {
        alt requests.try_recv() {
          some(get_count) { responses.send(copy count); }
          some(bytes(b)) {
            //#error("server: received %? bytes", b);
            count += b;
          }
          none { done = true; }
          _ { }
        }
    }
    responses.send(count);
    //#error("server exiting");
}

fn run(args: &[str]) {
    let (to_parent, from_child) = pipes::stream();
    let (to_child, from_parent) = pipes::stream();

    let to_child = shared_chan(to_child);

    let size = option::get(uint::from_str(args[1]));
    let workers = option::get(uint::from_str(args[2]));
    let num_bytes = 100;
    let start = std::time::precise_time_s();
    let mut worker_results = ~[];
    for uint::range(0u, workers) |i| {
        let builder = task::builder();
        vec::push(worker_results, task::future_result(builder));
        let to_child = to_child.clone();
        do task::run(builder) {
            for uint::range(0u, size / workers) |_i| {
                //#error("worker %?: sending %? bytes", i, num_bytes);
                to_child.send(bytes(num_bytes));
            }
            //#error("worker %? exiting", i);
        };
    }
    do task::spawn {
        server(from_parent, to_parent);
    }

    vec::iter(worker_results, |r| { future::get(r); } );
    //#error("sending stop message");
    to_child.send(stop);
    move!{to_child};
    let result = from_child.recv();
    let end = std::time::precise_time_s();
    let elapsed = end - start;
    io::stdout().write_str(#fmt("Count is %?\n", result));
    io::stdout().write_str(#fmt("Test took %? seconds\n", elapsed));
    let thruput = ((size / workers * workers) as float) / (elapsed as float);
    io::stdout().write_str(#fmt("Throughput=%f per sec\n", thruput));
    assert result == num_bytes * size;
}

fn main(args: ~[str]) {
    let args = if os::getenv("RUST_BENCH").is_some() {
        ~["", "1000000", "10000"]
    } else if args.len() <= 1u {
        ~["", "10000", "4"]
    } else {
        copy args
    };        

    #debug("%?", args);
    run(args);
}

// Treat a whole bunch of ports as one.
class box<T> {
    let mut contents: option<T>;
    new(+x: T) { self.contents = some(x); }

    fn swap(f: fn(+T) -> T) {
        let mut tmp = none;
        self.contents <-> tmp;
        self.contents = some(f(option::unwrap(tmp)));
    }

    fn unwrap() -> T {
        let mut tmp = none;
        self.contents <-> tmp;
        option::unwrap(tmp)
    }
}

class port_set<T: send> {
    let mut ports: ~[pipes::port<T>];

    new() { self.ports = ~[]; }

    fn add(+port: pipes::port<T>) {
        vec::push(self.ports, port)
    }

    fn try_recv() -> option<T> {
        let mut result = none;
        while result == none && self.ports.len() > 0 {
            let i = pipes::wait_many(self.ports.map(|p| p.header()));
            // dereferencing an unsafe pointer nonsense to appease the
            // borrowchecker.
            alt unsafe {(*ptr::addr_of(self.ports[i])).try_recv()} {
              some(m) {
                result = some(move!{m});
              }
              none {
                // Remove this port.
                let mut ports = ~[];
                self.ports <-> ports;
                vec::consume(ports,
                             |j, x| if i != j { vec::push(self.ports, x) });
              }
            }
        }
/*        
        while !done {
            do self.ports.swap |ports| {
                if ports.len() > 0 {
                    let old_len = ports.len();
                    let (_, m, ports) = pipes::select(ports);
                    alt m {
                      some(pipes::streamp::data(x, next)) {
                        result = some(move!{x});
                        done = true;
                        assert ports.len() == old_len - 1;
                        vec::append_one(ports, move!{next})
                      }
                      none {
                        //#error("pipe closed");
                        assert ports.len() == old_len - 1;
                        ports
                      }
                    }
                }
                else {
                    //#error("no more pipes");
                    done = true;
                    ~[]
                }
            }
        }
*/
        result
    }

    fn recv() -> T {
        option::unwrap(self.try_recv())
    }
}

impl private_methods/&<T: send> for pipes::port<T> {
    pure fn header() -> *pipes::packet_header unchecked {
        alt self.endp {
          some(endp) {
            endp.header()
          }
          none { fail "peeking empty stream" }
        }
    }
}

type shared_chan<T: send> = arc::exclusive<pipes::chan<T>>;

impl chan<T: send> for shared_chan<T> {
    fn send(+x: T) {
        let mut xx = some(x);
        do self.with |_c, chan| {
            let mut x = none;
            x <-> xx;
            chan.send(option::unwrap(x))
        }
    }
}

fn shared_chan<T:send>(+c: pipes::chan<T>) -> shared_chan<T> {
    arc::exclusive(c)
}