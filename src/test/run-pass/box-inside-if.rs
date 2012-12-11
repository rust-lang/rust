// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



// -*- rust -*-

fn some_box(x: int) -> @int { return @x; }

fn is_odd(n: int) -> bool { return true; }

fn length_is_even(vs: @int) -> bool { return true; }

fn foo(acc: int, n: int) {
    if is_odd(n) && length_is_even(some_box(1)) { error!("bloop"); }
}

fn main() { foo(67, 5); }
