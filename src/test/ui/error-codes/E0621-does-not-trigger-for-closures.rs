// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we give the generic E0495 when one of the free regions is
// bound in a closure (rather than suggesting a change to the signature
// of the closure, which is not specified in `foo` but rather in `invoke`).

// FIXME - This might be better as a UI test, but the finer details
// of the error seem to vary on different machines.
fn invoke<'a, F>(x: &'a i32, f: F) -> &'a i32
where F: FnOnce(&'a i32, &i32) -> &'a i32
{
    let y = 22;
    f(x, &y)
}

fn foo<'a>(x: &'a i32) {
    invoke(&x, |a, b| if a > b { a } else { b }); //~ ERROR E0495
}

fn main() {}
