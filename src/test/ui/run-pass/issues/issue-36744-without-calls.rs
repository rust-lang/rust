// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests for an LLVM abort when storing a lifetime-parametric fn into
// context that is expecting one that is not lifetime-parametric
// (i.e. has no `for <'_>`).

pub struct A<'a>(&'a ());
pub struct S<T>(T);

pub fn bad<'s>(v: &mut S<fn(A<'s>)>, y: S<for<'b> fn(A<'b>)>) {
    *v = y;
}

fn main() {}
