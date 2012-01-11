// -*- rust -*-
// xfail-pretty

/*
  A parallel version of fibonacci numbers.

  This version is meant mostly as a way of stressing and benchmarking
  the task system. It supports a lot of command-line arguments to
  control how it runs.

*/

use std;

import std::{time, io, getopts};
import io::writer_util;
import int::range;
import comm::port;
import comm::chan;
import comm::send;
import comm::recv;

import core::result;
import result::{ok, err};

fn fib(n: int) -> int {
    fn pfib(c: chan<int>, n: int) {
        if n == 0 {
            send(c, 0);
        } else if n <= 2 {
            send(c, 1);
        } else {
            let p = port();
            let ch = chan(p);
            task::spawn {|| pfib(ch, n - 1); };
            task::spawn {|| pfib(ch, n - 2); };
            send(c, recv(p) + recv(p));
        }
    }

    let p = port();
    let ch = chan(p);
    let t = task::spawn {|| pfib(ch, n); };
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
        tasks += [task::spawn_joinable {|| stress_task(i); }];
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
