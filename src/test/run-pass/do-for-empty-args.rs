// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-reformat
// Testing various forms of `do` and `for` with empty arg lists

fn f(f: &fn() -> bool) -> bool {
    true
}

pub fn main() {
    do f() || { true };
    do f() { true };
    do f || { true };
    do f { true };
    for f() || { }
    for f() { }
    for f || { }
    for f { }
}
