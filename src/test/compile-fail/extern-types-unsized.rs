// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure extern types are !Sized and !DynSized.

#![feature(extern_types)]

extern {
    type A;
}

fn assert_sized<T>() { }
fn assert_dynsized<T: ?Sized>() { }

fn main() {
    assert_sized::<A>();
    //~^ ERROR the trait bound `A: std::marker::Sized` is not satisfied

    assert_dynsized::<A>();
    //~^ ERROR the trait bound `A: std::marker::DynSized` is not satisfied
}
