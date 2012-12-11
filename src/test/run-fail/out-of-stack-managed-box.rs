// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test iloops with optimizations on

// NB: Not sure why this works. I expect the box argument to leak when
// we run out of stack. Maybe the box annihilator works it out?

// error-pattern:ran out of stack
fn main() {
    eat(move @0);
}

fn eat(
    +a: @int
) {
    eat(move a)
}
