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
        ~[~"", ~"10000000"]
    } else if args.len() <= 1u {
        ~[~"", ~"100000"]
    } else {
        args
    };

    let n = uint::from_str(args[1]).get();

    for uint::range(0u, n) |i| {
        let x = uint::to_str(i);
        info!(x);
    }
}
