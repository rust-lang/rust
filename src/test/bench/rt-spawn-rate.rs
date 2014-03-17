// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task::spawn;
use std::os;
use std::uint;

// Very simple spawn rate test. Spawn N tasks that do nothing and
// return.

fn main() {

    let args = os::args();
    let n = if args.len() == 2 {
        from_str::<uint>(args[1]).unwrap()
    } else {
        100000
    };

    for _ in range(0, n) {
        spawn(proc() {});
    }

}
