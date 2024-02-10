//! This test is reporting the wrong error. We need
//! more inherent associated type tests that use opaque types
//! in general. Some variant of this test should compile successfully.
// known-bug: unknown
// edition:2018

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

use std::future::Future;

struct Foo<'a>(&'a mut ());

impl Foo<'_> {
    type Fut<'a> = impl Future<Output = ()>;

    fn make_fut<'a>(&'a self) -> Self::Fut<'a> {
        async { () }
    }
}

fn main() {}
