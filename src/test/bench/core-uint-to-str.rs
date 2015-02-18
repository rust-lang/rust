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

fn main() {
    let args = env::args();
    let args = if env::var_os("RUST_BENCH").is_some() {
        vec!("".to_string(), "10000000".to_string())
    } else if args.len() <= 1 {
        vec!("".to_string(), "100000".to_string())
    } else {
        args.collect()
    };

    let n = args[1].parse().unwrap();

    for i in 0..n {
        let x = i.to_string();
        println!("{}", x);
    }
}
