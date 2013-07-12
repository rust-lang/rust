// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

use std::int;
use std::io;
use std::os;

fn ack(m: int, n: int) -> int {
    if m == 0 {
        return n + 1
    } else {
        if n == 0 {
            return ack(m - 1, 1);
        } else {
            return ack(m - 1, ack(m, n - 1));
        }
    }
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"12"]
    } else if args.len() <= 1u {
        ~[~"", ~"8"]
    } else {
        args
    };
    let n = int::from_str(args[1]).get();
    io::println(fmt!("Ack(3,%d): %d\n", n, ack(3, n)));
}
