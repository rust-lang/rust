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
use std::uint;

fn main() {
    let args = os::args();
    let args = if os::getenv("RUST_BENCH").is_some() {
        vec!("".to_string(), "10000000".to_string())
    } else if args.len() <= 1u {
        vec!("".to_string(), "100000".to_string())
    } else {
        args.into_iter().collect()
    };

    let n = from_str::<uint>(args[1].as_slice()).unwrap();

    for i in range(0u, n) {
        let x = i.to_string();
        println!("{}", x);
    }
}
