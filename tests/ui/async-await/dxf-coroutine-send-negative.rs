//@ edition: 2024
//@ compile-flags: -Znext-solver -Zdxf

#![allow(dead_code)]

use std::marker::PhantomData;

struct NeedsMerge<'a, 'b> {
    _raw: *const (),
    _inv_a: PhantomData<fn(&'a ()) -> &'a ()>,
    _inv_b: PhantomData<fn(&'b ()) -> &'b ()>,
}

unsafe impl<'a> Send for NeedsMerge<'a, 'a> {}

async fn yield_point() {}

async fn use_phantom(data1: &(), data2: &()) {
    let val = NeedsMerge {
        _raw: std::ptr::null(),
        _inv_a: PhantomData, // '?p
        _inv_b: PhantomData, // '?q
    };
    yield_point().await;
    drop(val);
}

fn assert_send(_: impl Send) {}

fn main() {
    assert_send(use_phantom(&(), &()));
    //~^ ERROR
    //~| ERROR
    // The reason is NLL cannot deduce from the coroutine body that the two
    // regions are actually the same.
    // The two existential lifetimes '?p and '?q are independent to one another.
    // They are left unconstrained by the implicit lifetime generics.
    // The analysis is still sound, because validity of the coroutine body
    // may depend on the fact that the two regions are independent.
}
