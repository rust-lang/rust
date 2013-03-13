// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait bar { fn dup() -> Self; fn blah<X>(); }
impl bar for int { fn dup() -> int { self } fn blah<X>() {} }
impl bar for uint { fn dup() -> uint { self } fn blah<X>() {} }

fn main() {
    10i.dup::<int>(); //~ ERROR does not take type parameters
    10i.blah::<int, int>(); //~ ERROR incorrect number of type parameters
    (@10 as @bar).dup(); //~ ERROR contains a self-type
}
