// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Deref;

trait PointerFamily<U> {
    type Pointer<T>: Deref<Target = T>;
    //~^ ERROR generic associated types are unstable
    type Pointer2<T>: Deref<Target = T> where T: Clone, U: Clone;
    //~^ ERROR generic associated types are unstable
    //~| ERROR where clauses on associated types are unstable
}

struct Foo;
impl PointerFamily<u32> for Foo {
    type Pointer<usize> = Box<usize>;
    //~^ ERROR generic associated types are unstable
    type Pointer2<u32> = Box<u32>;
    //~^ ERROR generic associated types are unstable
}

trait Bar {
    type Assoc where Self: Sized;
    //~^ ERROR where clauses on associated types are unstable
}


fn main() {}
