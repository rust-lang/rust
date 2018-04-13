// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test
// compile-pass

#![feature(infer_outlives_requirements)]
// Outlives requirementes are inferred (RFC 2093)

trait MakeRef<'a> {
    type Type;
}
impl<'a, T> MakeRef<'a> for Vec<T>
where T: 'a,
{
    type Type = &'a T;
}
// explicit-impl: T: 'a
struct Foo<'a, T> {
    foo: <Vec<T> as MakeRef<'a>>::Type,
}

fn main() {}
