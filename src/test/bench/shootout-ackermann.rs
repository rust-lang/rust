// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
    let args = if os::getenv("RUST_BENCH").is_some() {
        vec!("".to_owned(), "12".to_owned())
    } else if args.len() <= 1u {
        vec!("".to_owned(), "8".to_owned())
    } else {
        args.move_iter().collect()
    };
    let n = from_str::<int>(args.get(1).as_slice()).unwrap();
    println!("Ack(3,{}): {}\n", n, ack(3, n));
}
