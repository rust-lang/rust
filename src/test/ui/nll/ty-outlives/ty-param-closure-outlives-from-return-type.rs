// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Znll -Zborrowck=mir -Zverbose

#![allow(warnings)]
#![feature(dyn_trait)]
#![feature(rustc_attrs)]

use std::fmt::Debug;

fn with_signature<'a, T, F>(x: Box<T>, op: F) -> Box<dyn Debug + 'a>
    where F: FnOnce(Box<T>) -> Box<dyn Debug + 'a>
{
    op(x)
}

#[rustc_regions]
fn no_region<'a, T>(x: Box<T>) -> Box<dyn Debug + 'a>
where
    T: Debug,
{
    // Here, the closure winds up being required to prove that `T:
    // 'a`.  In principle, it could know that, except that it is
    // type-checked in a fully generic way, and hence it winds up with
    // a propagated requirement that `T: '_#2`, where `'_#2` appears
    // in the return type. The caller makes the mapping from `'_#2` to
    // `'a` (and subsequently reports an error).

    with_signature(x, |y| y)
    //~^ WARNING not reporting region error due to -Znll
    //~| ERROR the parameter type `T` may not live long enough
}

fn correct_region<'a, T>(x: Box<T>) -> Box<Debug + 'a>
where
    T: 'a + Debug,
{
    x
}

fn wrong_region<'a, 'b, T>(x: Box<T>) -> Box<Debug + 'a>
where
    T: 'b + Debug,
{
    x
    //~^ WARNING not reporting region error due to -Znll
    //~| ERROR the parameter type `T` may not live long enough
}

fn outlives_region<'a, 'b, T>(x: Box<T>) -> Box<Debug + 'a>
where
    T: 'b + Debug,
    'b: 'a,
{
    x
}

fn main() {}
