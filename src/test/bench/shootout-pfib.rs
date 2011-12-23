// -*- rust -*-
// xfail-pretty

/*
  A parallel version of fibonacci numbers.

  This version is meant mostly as a way of stressing and benchmarking
  the task system. It supports a lot of command-line arguments to
  control how it runs.

*/

use std;

import vec;
import uint;
import std::time;
import str;
import int::range;
import std::io;
import std::getopts;
import task;
import u64;
import comm;
import comm::port;
import comm::chan;
import comm::send;
import comm::recv;

import core::result;
import result::{ok, err};

fn fib(n: int) -> int {
    fn pfib(args: (chan<int>, int)) {
        let (c, n) = args;
        if n == 0 {
            send(c, 0);
        } else if n <= 2 {
            send(c, 1);
        } else {
            let p = port();

            let t1 = task::spawn((chan(p), n - 1), pfib);
            let t2 = task::spawn((chan(p), n - 2), pfib);

            send(c, recv(p) + recv(p));
        }
    }

    let p = port();
    let t = task::spawn((chan(p), n), pfib);
    ret recv(p);
}

type config = {stress: bool};

fn parse_opts(argv: [str]) -> config {
    let opts = [getopts::optflag("stress")];

    let opt_args = vec::slice(argv, 1u, vec::len(argv));


    alt getopts::getopts(opt_args, opts) {
      ok(m) { ret {stress: getopts::opt_present(m, "stress")} }
      err(_) { fail; }
    }
}

fn stress_task(&&id: int) {
    let i = 0;
    while true {
        let n = 15;
        assert (fib(n) == fib(n));
        i += 1;
        #error("%d: Completed %d iterations", id, i);
    }
}

fn stress(num_tasks: int) {
    let tasks = [];
    range(0, num_tasks) {|i|
        tasks += [task::spawn_joinable(copy i, stress_task)];
    }
    for t in tasks { task::join(t); }
}

fn main(argv: [str]) {
    if vec::len(argv) == 1u {
        assert (fib(8) == 21);
        log(debug, fib(8));
    } else {
        // Interactive mode! Wooo!!!!
        let opts = parse_opts(argv);


        if opts.stress {
            stress(2);
        } else {
            let max = uint::parse_buf(str::bytes(argv[1]), 10u) as int;

            let num_trials = 10;

            let out = io::stdout();

            range(1, max + 1) {|n|
                range(0, num_trials) {|i|
                    let start = time::precise_time_ns();
                    let fibn = fib(n);
                    let stop = time::precise_time_ns();

                    let elapsed = stop - start;

                    out.write_line(#fmt["%d\t%d\t%s", n, fibn,
                                        u64::str(elapsed)]);
                }
            }
        }
    }
}
