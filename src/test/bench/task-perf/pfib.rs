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
import std::str;
import std::int::range;
import std::io;

fn recv[T](&port[T] p) -> T {
    let T x;
    p |> x;
    ret x;
}

fn fib(int n) -> int {
    fn pfib(chan[int] c, int n) {
        if (n == 0) {
            c <| 0;
        }
        else if (n <= 2) {
            c <| 1;
        }
        else {
            let port[int] p = port();
      
            auto t1 = spawn pfib(chan(p), n - 1);
            auto t2 = spawn pfib(chan(p), n - 2);

            c <| recv(p) + recv(p);
        }
    }

    let port[int] p = port();
    auto t = spawn pfib(chan(p), n);
    ret recv(p);
}

fn main(vec[str] argv) {
    if(vec::len(argv) == 1u) {
        assert (fib(8) == 21);
        assert (fib(15) == 610);
        log fib(8);
        log fib(15);
    }
    else {
        // Interactive mode! Wooo!!!!

        auto max = uint::parse_buf(str::bytes(argv.(1)), 10u) as int;

        auto num_trials = 10;

        auto out = io::stdout();

        for each(int n in range(1, max + 1)) {
            for each(int i in range(0, num_trials)) {
                auto start = time::precise_time_ns();
                auto fibn = fib(n);
                auto stop = time::precise_time_ns();

                auto elapsed = (stop - start) as int;
            
                out.write_line(#fmt("%d\t%d\t%d", n, fibn, elapsed));
            }
        }

    }
}
