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
use std::task;
use std::uint;

fn f(n: uint) {
    let mut i = 0u;
    while i < n {
        task::try(proc() g());
        i += 1u;
    }
}

fn g() { }

fn main() {
    let args = os::args();
    let args = if os::getenv("RUST_BENCH").is_some() {
        vec!(~"", ~"400")
    } else if args.len() <= 1u {
        vec!(~"", ~"10")
    } else {
        args.move_iter().collect()
    };
    let n = from_str::<uint>(*args.get(1)).unwrap();
    let mut i = 0u;
    while i < n { task::spawn(proc() f(n) ); i += 1u; }
}
