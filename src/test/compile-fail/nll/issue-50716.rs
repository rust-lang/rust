// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Regression test for the issue #50716: NLL ignores lifetimes bounds
// derived from `Sized` requirements

trait A {
    type X: ?Sized;
}

fn foo<'a, T: 'static>(s: Box<<&'a T as A>::X>)
where
    for<'b> &'b T: A,
    <&'static T as A>::X: Sized
{
    let _x = *s; //~ ERROR mismatched types [E0308]
}

fn main() {}
