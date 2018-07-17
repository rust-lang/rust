// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let bar: fn(&mut u32) = |_| {};

    fn foo(x: Box<Fn(&i32)>) {}
    let bar = Box::new(|x: &i32| {}) as Box<Fn(_)>;
    foo(bar); //~ ERROR mismatched types
    //~| expected concrete lifetime, found bound lifetime parameter
}
