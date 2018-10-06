// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(dyn_trait)]
#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]

trait Trait<'x, T> where T: 'x {
}

#[rustc_outlives]
struct Foo<'a, A> //~ ERROR 19:1: 22:2: rustc_outlives
{
    foo: Box<dyn Trait<'a, A>>
}

fn main() {}
