// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #2783

fn foo(f: ||) { f() }

fn main() {
    ~"" || 42; //~ ERROR binary operation || cannot be applied to type
    foo || {}; //~ ERROR binary operation || cannot be applied to type
    //~^ NOTE did you forget the `do` keyword for the call?
}
