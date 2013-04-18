// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test

// compile-flags:-Z extra-debug-info
// debugger:set print pretty off
// debugger:break 29
// debugger:run
// debugger:print pair
// check:$1 = {x = 1, y = 2}
// debugger:print pair.x
// check:$2 = 1
// debugger:print pair.y
// check:$3 = 2

struct Pair {
    x: int,
    y: int
}

fn main() {
    let pair = Pair { x: 1, y: 2 };
    let _z = ();
}
