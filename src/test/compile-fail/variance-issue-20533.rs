// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #20533. At some point, only 1 out of the
// 3 errors below were being reported.

use std::marker::PhantomData;

fn foo<'a, T>(_x: &'a T) -> PhantomData<&'a ()> {
    PhantomData
}

struct Wrap<T>(T);

fn bar<'a, T>(_x: &'a T) -> Wrap<PhantomData<&'a ()>> {
    Wrap(PhantomData)
}

struct Baked<'a>(PhantomData<&'a ()>);

fn baz<'a, T>(_x: &'a T) -> Baked<'a> {
    Baked(PhantomData)
}

struct AffineU32(u32);

fn main() {
    {
        let a = AffineU32(1_u32);
        let x = foo(&a);
        drop(a); //~ ERROR cannot move out of `a`
        drop(x);
    }
    {
        let a = AffineU32(1_u32);
        let x = bar(&a);
        drop(a); //~ ERROR cannot move out of `a`
        drop(x);
    }
    {
        let a = AffineU32(1_u32);
        let x = baz(&a);
        drop(a); //~ ERROR cannot move out of `a`
        drop(x);
    }
}

