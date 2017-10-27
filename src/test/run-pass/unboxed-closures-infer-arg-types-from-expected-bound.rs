// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are able to infer that the type of `x` is `isize` based
// on the expected type from the object.

// pretty-expanded FIXME #23616

pub trait ToPrimitive {
    fn to_int(&self) {}
}

impl ToPrimitive for isize {}
impl ToPrimitive for i32 {}
impl ToPrimitive for usize {}

fn doit<T,F>(val: T, f: &F)
    where F : Fn(T)
{
    f(val)
}

pub fn main() {
    doit(0, &|x /*: isize*/ | { x.to_int(); });
}
