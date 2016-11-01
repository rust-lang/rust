// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test used to be part of a run-pass test, but revised outlives
// rule means that it no longer compiles.

#![allow(unused_variables)]

trait Trait<'a> {
    fn long(&'a self) -> isize;
    fn short<'b>(&'b self) -> isize;
}

fn object_invoke1<'d>(x: &'d Trait<'d>) -> (isize, isize) { loop { } }

trait MakerTrait {
    fn mk() -> Self;
}

fn make_val<T:MakerTrait>() -> T {
    MakerTrait::mk()
}

impl<'t> MakerTrait for Box<Trait<'t>+'static> {
    fn mk() -> Box<Trait<'t>+'static> { loop { } }
}

pub fn main() {
    let m : Box<Trait+'static> = make_val();
    assert_eq!(object_invoke1(&*m), (4,5));
    //~^ NOTE borrow occurs here

    // the problem here is that the full type of `m` is
    //
    //   Box<Trait<'m>+'static>
    //
    // Here `'m` must be exactly the lifetime of the variable `m`.
    // This is because of two requirements:
    // 1. First, the basic type rules require that the
    //    type of `m`'s value outlives the lifetime of `m`. This puts a lower
    //    bound `'m`.
    //
    // 2. Meanwhile, the signature of `object_invoke1` requires that
    //    we create a reference of type `&'d Trait<'d>` for some `'d`.
    //    `'d` cannot outlive `'m`, so that forces the lifetime to be `'m`.
    //
    // This then conflicts with the dropck rules, which require that
    // the type of `m` *strictly outlives* `'m`. Hence we get an
    // error.
}
//~^ ERROR `*m` does not live long enough
//~| NOTE `*m` dropped here while still borrowed
//~| NOTE values in a scope are dropped in the opposite order they are created

