// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Simplified regression test for #30438, inspired by arielb1.

trait Trait { type Out; }

struct Test<'a> { s: &'a str }

fn silly<'y, 'z>(_s: &'y Test<'z>) -> &'y <Test<'z> as Trait>::Out where 'z: 'static {
    let x = Test { s: "this cannot last" };
    &x
    //~^ ERROR: `x` does not live long enough
}

impl<'b> Trait for Test<'b> { type Out = Test<'b>; }

fn main() {
    let orig = Test { s: "Hello World" };
    let r = silly(&orig);
    println!("{}", orig.s); // OK since `orig` is valid
    println!("{}", r.s); // Segfault (method does not return a sane value)
}
