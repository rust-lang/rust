// -*- rust -*-
// xfail-pretty

/*
  A parallel version of fibonacci numbers.

  This version is meant mostly as a way of stressing and benchmarking
  the task system. It supports a lot of command-line arguments to
  control how it runs.

*/

#[legacy_modes];

extern mod std;

use std::{time, getopts};
use io::WriterUtil;
use int::range;
use pipes::Port;
use pipes::Chan;
use pipes::send;
use pipes::recv;

use core::result;
use result::{Ok, Err};

fn fib(n: int) -> int {
    fn pfib(c: Chan<int>, n: int) {
        if n == 0 {
            c.send(0);
        } else if n <= 2 {
            c.send(1);
        } else {
            let p = pipes::PortSet();
            let ch = p.chan();
            task::spawn(|| pfib(ch, n - 1) );
            let ch = p.chan();
            task::spawn(|| pfib(ch, n - 2) );
            c.send(p.recv() + p.recv());
        }
    }

    let (ch, p) = pipes::stream();
    let t = task::spawn(|| pfib(ch, n) );
    p.recv()
}

type config = {stress: bool};

fn parse_opts(argv: ~[~str]) -> config {
    let opts = ~[getopts::optflag(~"stress")];

    let opt_args = vec::slice(argv, 1u, vec::len(argv));

    match getopts::getopts(opt_args, opts) {
      Ok(m) => { return {stress: getopts::opt_present(m, ~"stress")} }
      Err(_) => { fail; }
    }
}

fn stress_task(&&id: int) {
    let mut i = 0;
    loop {
        let n = 15;
        assert (fib(n) == fib(n));
        i += 1;
        error!("%d: Completed %d iterations", id, i);
    }
}

fn stress(num_tasks: int) {
    let mut results = ~[];
    for range(0, num_tasks) |i| {
        do task::task().future_result(|+r| {
            results.push(r);
        }).spawn {
            stress_task(i);
        }
    }
    for results.each |r| { future::get(r); }
}

fn main(++args: ~[~str]) {
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"20"]
    } else if args.len() <= 1u {
        ~[~"", ~"8"]
    } else {
        args
    };

    let opts = parse_opts(args);

    if opts.stress {
        stress(2);
    } else {
        let max = uint::parse_bytes(str::to_bytes(args[1]),
                                                10u).get() as int;

        let num_trials = 10;

        let out = io::stdout();

        for range(1, max + 1) |n| {
            for range(0, num_trials) |i| {
                let start = time::precise_time_ns();
                let fibn = fib(n);
                let stop = time::precise_time_ns();

                let elapsed = stop - start;

                out.write_line(fmt!("%d\t%d\t%s", n, fibn,
                                    u64::str(elapsed)));
            }
        }
    }
}
