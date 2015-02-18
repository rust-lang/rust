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
use std::thread;

fn f(n: usize) {
    let mut i = 0;
    while i < n {
        let _ = thread::spawn(move|| g()).join();
        i += 1;
    }
}

fn g() { }

fn main() {
    let args = env::args();
    let args = if env::var_os("RUST_BENCH").is_some() {
        vec!("".to_string(), "400".to_string())
    } else if args.len() <= 1 {
        vec!("".to_string(), "10".to_string())
    } else {
        args.collect()
    };
    let n = args[1].parse().unwrap();
    let mut i = 0;
    while i < n { thread::spawn(move|| f(n) ); i += 1; }
}
