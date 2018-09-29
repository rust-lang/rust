// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![allow(unused)]
#![deny(explicit_outlives_requirements)]

// A case where we can't infer the outlives requirement. Example copied from
// RFC 2093.
// (https://rust-lang.github.io/rfcs/2093-infer-outlives.html
// #where-explicit-annotations-would-still-be-required)


trait MakeRef<'a> {
    type Type;
}

impl<'a, T> MakeRef<'a> for Vec<T>
    where T: 'a  // still required
{
    type Type = &'a T;
}


struct Foo<'a, T>
    where T: 'a  // still required, not inferred from `field`
{
    field: <Vec<T> as MakeRef<'a>>::Type
}


fn main() {}
