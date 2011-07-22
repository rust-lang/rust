// -*- rust -*-

/*
  A parallel version of fibonacci numbers.
*/

use std;

import std::vec;
import std::uint;
import std::time;
import std::str;

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
        //assert (fib(15) == 610);
        log fib(8);
        //log fib(15);
    }
    else {
        // Interactive mode! Wooo!!!!

        auto n = uint::parse_buf(str::bytes(argv.(1)), 10u) as int;
        auto start = time::precise_time_ns();
        auto fibn = fib(n);
        auto stop = time::precise_time_ns();

        assert(stop >= start);

        auto elapsed = stop - start;
        auto us_task = elapsed / (fibn as u64) / (1000 as u64);

        log_err #fmt("Determined that fib(%d) = %d in %d%d ns (%d us / task)",
                     n, fibn,
                     (elapsed / (1000000 as u64)) as int,
                     (elapsed % (1000000 as u64)) as int,
                     us_task as int);
    }
}
