// -*- rust -*-

/*
  A parallel version of fibonacci numbers.

  This version is meant mostly as a way of stressing and benchmarking
  the task system. It supports a lot of command-line arguments to
  control how it runs.

*/

use std;

import std::vec;
import std::ivec;
import std::uint;
import std::time;
import std::str;
import std::int::range;
import std::ioivec;
import std::getopts;
import std::task;
import std::u64;

fn recv[T](p: &port[T]) -> T { let x: T; p |> x; ret x; }

fn fib(n: int) -> int {
    fn pfib(c: chan[int], n: int) {
        if n == 0 {
            c <| 0;
        } else if (n <= 2) {
            c <| 1;
        } else {
            let p = port();

            let t1 = spawn pfib(chan(p), n - 1);
            let t2 = spawn pfib(chan(p), n - 2);

            c <| recv(p) + recv(p);
        }
    }

    let p = port();
    let t = spawn pfib(chan(p), n);
    ret recv(p);
}

type config = {stress: bool};

fn parse_opts(argv: vec[str]) -> config {
    let opts = [getopts::optflag("stress")];

    let opt_args = vec::slice(argv, 1u, vec::len(argv));


    alt getopts::getopts(opt_args, opts) {
      getopts::success(m) { ret {stress: getopts::opt_present(m, "stress")} }
      getopts::failure(_) { fail; }
    }
}

fn stress_task(id: int) {
    let i = 0;
    while true {
        let n = 15;
        assert (fib(n) == fib(n));
        i += 1;
        log_err #fmt("%d: Completed %d iterations", id, i);
    }
}

fn stress(num_tasks: int) {
    let tasks = [];
    for each i: int  in range(0, num_tasks) {
        tasks += [spawn stress_task(i)];
    }
    for each i: int  in range(0, num_tasks) { task::join(tasks.(i)); }
}

fn main(argv: vec[str]) {
    if vec::len(argv) == 1u {
        assert (fib(8) == 21);
        log fib(8);
    } else {
        // Interactive mode! Wooo!!!!
        let opts = parse_opts(argv);


        if opts.stress {
            stress(2);
        } else {
            let max = uint::parse_buf(ivec::to_vec(str::bytes(argv.(1))),
                                      10u) as int;

            let num_trials = 10;

            let out = ioivec::stdout();


            for each n: int  in range(1, max + 1) {
                for each i: int  in range(0, num_trials) {
                    let start = time::precise_time_ns();
                    let fibn = fib(n);
                    let stop = time::precise_time_ns();

                    let elapsed = stop - start;

                    out.write_line(#fmt("%d\t%d\t%s", n, fibn,
                                        u64::str(elapsed)));
                }
            }
        }
    }
}