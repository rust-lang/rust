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
// debugger:break 20
// debugger:run
// debugger:print t
// check:$1 = {4, 5.5, true}

fn main() {
    let t = (4, 5.5, true);
    let _z = ();
}
