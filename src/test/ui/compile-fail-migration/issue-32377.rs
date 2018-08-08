// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;
use std::marker::PhantomData;

trait Foo {
    type Error;
}

struct Bar<U: Foo> {
    stream: PhantomData<U::Error>,
}

fn foo<U: Foo>(x: [usize; 2]) -> Bar<U> {
    unsafe { mem::transmute(x) }
    //~^ ERROR transmute called with types of different sizes
}

fn main() {}
