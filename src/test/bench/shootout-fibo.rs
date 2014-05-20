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

fn fib(n: int) -> int {
    if n < 2 {
        return 1;
    } else {
        return fib(n - 1) + fib(n - 2);
    }
}

fn main() {
    let args = os::args();
    let args = if os::getenv("RUST_BENCH").is_some() {
        vec!("".to_owned(), "40".to_owned())
    } else if args.len() <= 1u {
        vec!("".to_owned(), "30".to_owned())
    } else {
        args.move_iter().collect()
    };
    let n = from_str::<int>(args.get(1).as_slice()).unwrap();
    println!("{}\n", fib(n));
}
