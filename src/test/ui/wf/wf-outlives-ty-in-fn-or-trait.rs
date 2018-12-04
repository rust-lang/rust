// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(dead_code)]

trait Trait<'a, T> {
    type Out;
}

impl<'a, T> Trait<'a, T> for usize {
    type Out = &'a fn(T); //~ ERROR `T` may not live long enough
}

struct Foo<'a,T> {
    f: &'a fn(T),
}

trait Baz<T> { }

impl<'a, T> Trait<'a, T> for u32 {
    type Out = &'a Baz<T>; //~ ERROR `T` may not live long enough
}

fn main() { }

