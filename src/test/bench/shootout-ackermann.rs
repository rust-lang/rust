// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;

fn ack(m: i64, n: i64) -> i64 {
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
    let mut args = env::args();
    let args = if env::var_os("RUST_BENCH").is_some() {
        vec!("".to_string(), "12".to_string())
    } else if args.len() <= 1 {
        vec!("".to_string(), "8".to_string())
    } else {
        args.collect()
    };
    let n = args[1].parse().unwrap();
    println!("Ack(3,{}): {}\n", n, ack(3, n));
}
