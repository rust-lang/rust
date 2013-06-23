// -*- rust -*-
// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


/*
  A parallel version of fibonacci numbers.

  This version is meant mostly as a way of stressing and benchmarking
  the task system. It supports a lot of old command-line arguments to
  control how it runs.

*/

extern mod extra;

use extra::{time, getopts};
use std::comm::*;
use std::int::range;
use std::io::WriterUtil;
use std::io;
use std::os;
use std::result::{Ok, Err};
use std::str;
use std::task;
use std::u64;
use std::uint;
use std::vec;

fn fib(n: int) -> int {
    fn pfib(c: &Chan<int>, n: int) {
        if n == 0 {
            c.send(0);
        } else if n <= 2 {
            c.send(1);
        } else {
            let p = PortSet::new();
            let ch = p.chan();
            task::spawn(|| pfib(&ch, n - 1) );
            let ch = p.chan();
            task::spawn(|| pfib(&ch, n - 2) );
            c.send(p.recv() + p.recv());
        }
    }

    let (p, ch) = stream();
    let _t = task::spawn(|| pfib(&ch, n) );
    p.recv()
}

struct Config {
    stress: bool
}

fn parse_opts(argv: ~[~str]) -> Config {
    let opts = ~[getopts::optflag(~"stress")];

    let opt_args = vec::slice(argv, 1, argv.len());

    match getopts::getopts(opt_args, opts) {
      Ok(ref m) => {
          return Config {stress: getopts::opt_present(m, ~"stress")}
      }
      Err(_) => { fail!(); }
    }
}

fn stress_task(id: int) {
    let mut i = 0;
    loop {
        let n = 15;
        assert_eq!(fib(n), fib(n));
        i += 1;
        error!("%d: Completed %d iterations", id, i);
    }
}

fn stress(num_tasks: int) {
    let mut results = ~[];
    for range(0, num_tasks) |i| {
        let mut builder = task::task();
        builder.future_result(|r| results.push(r));
        do builder.spawn {
            stress_task(i);
        }
    }
    for results.iter().advance |r| {
        r.recv();
    }
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"20"]
    } else if args.len() <= 1u {
        ~[~"", ~"8"]
    } else {
        args
    };

    let opts = parse_opts(copy args);

    if opts.stress {
        stress(2);
    } else {
        let max = uint::parse_bytes(args[1].as_bytes(), 10u).get() as int;

        let num_trials = 10;

        let out = io::stdout();

        for range(1, max + 1) |n| {
            for range(0, num_trials) |_i| {
                let start = time::precise_time_ns();
                let fibn = fib(n);
                let stop = time::precise_time_ns();

                let elapsed = stop - start;

                out.write_line(fmt!("%d\t%d\t%s", n, fibn,
                                    u64::to_str(elapsed)));
            }
        }
    }
}
