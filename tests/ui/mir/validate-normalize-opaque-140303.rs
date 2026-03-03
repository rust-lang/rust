//@ compile-flags: -Zvalidate-mir
//@ edition: 2021
//
// Regression test for #140303.
// MIR validation should not ICE when normalizing opaque types
// that involve incomplete trait impls.

use std::future::Future;

async fn create_task() -> impl Sized {
    bind(documentation)
}

async fn documentation() {}

fn bind<F>(_filter: F) -> impl Sized
where
    F: FilterBase,
{
    || -> <F as FilterBase>::Assoc { panic!() }
}

trait FilterBase {
    type Assoc;
}

impl<F, R> FilterBase for F
//~^ ERROR not all trait items implemented, missing: `Assoc`
where
    F: Fn() -> R,
    R: Future,
    R: Send,
{
}

fn main() {}
