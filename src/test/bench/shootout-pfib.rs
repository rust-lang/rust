// -*- rust -*-

/*
  A parallel version of fibonacci numbers.

  This version is meant mostly as a way of stressing and benchmarking
  the task system. It supports a lot of command-line arguments to
  control how it runs.

*/

use std;

import std::vec;
import std::uint;
import std::time;
import std::istr;
import std::int::range;
import std::io;
import std::getopts;
import std::task;
import std::u64;
import std::comm;
import std::comm::port;
import std::comm::chan;
import std::comm::send;
import std::comm::recv;

fn fib(n: int) -> int {
    fn pfib(c: chan<int>, n: int) {
        if n == 0 {
            send(c, 0);
        } else if n <= 2 {
            send(c, 1);
        } else {
            let p = port();

            let t1 = task::spawn(bind pfib(chan(p), n - 1));
            let t2 = task::spawn(bind pfib(chan(p), n - 2));

            send(c, recv(p) + recv(p));
        }
    }

    let p = port();
    let t = task::spawn(bind pfib(chan(p), n));
    ret recv(p);
}

type config = {stress: bool};

fn parse_opts(argv: &[istr]) -> config {
    let opts = [getopts::optflag(~"stress")];

    let opt_args = vec::slice(argv, 1u, vec::len(argv));


    alt getopts::getopts(opt_args, opts) {
      getopts::success(m) { ret {stress: getopts::opt_present(m, ~"stress")} }
      getopts::failure(_) { fail; }
    }
}

fn stress_task(id: int) {
    let i = 0;
    while true {
        let n = 15;
        assert (fib(n) == fib(n));
        i += 1;
        log_err #ifmt["%d: Completed %d iterations", id, i];
    }
}

fn stress(num_tasks: int) {
    let tasks = [];
    for each i: int in range(0, num_tasks) {
        tasks += [task::spawn_joinable(bind stress_task(i))];
    }
    for t in tasks { task::join(t); }
}

fn main(argv: [str]) {
    let argv = istr::from_estrs(argv);
    if vec::len(argv) == 1u {
        assert (fib(8) == 21);
        log fib(8);
    } else {
        // Interactive mode! Wooo!!!!
        let opts = parse_opts(argv);


        if opts.stress {
            stress(2);
        } else {
            let max = uint::parse_buf(istr::bytes(argv[1]), 10u) as int;

            let num_trials = 10;

            let out = io::stdout();

            for each n: int in range(1, max + 1) {
                for each i: int in range(0, num_trials) {
                    let start = time::precise_time_ns();
                    let fibn = fib(n);
                    let stop = time::precise_time_ns();

                    let elapsed = stop - start;

                    out.write_line(
                            #ifmt["%d\t%d\t%s", n, fibn,
                                 u64::str(elapsed)]);
                }
            }
        }
    }
}
