//@ edition: 2024
//@ revisions: no_dxf dxf
//@[no_dxf] compile-flags: -Znext-solver -Zassumptions-on-binders
//@[no_dxf] check-pass
//@[dxf] compile-flags: -Znext-solver -Zdxf

#![allow(dead_code)]

use std::marker::PhantomData;

struct NeedsMerge<'a, 'b> {
    _raw: *const (),
    _inv_a: PhantomData<fn(&'a ()) -> &'a ()>,
    _inv_b: PhantomData<fn(&'b ()) -> &'b ()>,
}

unsafe impl<'a> Send for NeedsMerge<'a, 'a> {}

async fn inner(data: &()) {
    let val = NeedsMerge {
        _raw: std::ptr::null(),
        _inv_a: PhantomData, // '?long
        _inv_b: PhantomData, // '?short
    };
    std::future::pending::<()>().await;
    drop(val);
}

async fn outer(data: &()) {
    // The future returned by inner(data) is held across a yield.
    inner(data).await;
}

fn assert_send(_: impl Send) {}

fn main() {
    assert_send(outer(&()));
    //[dxf]~^ ERROR
    //[dxf]~| ERROR
    // I think -Zassumptions-on-binders might be unsound here.
    // I can see that '?long outlives '?short, which -Zdxf can compute by
    // maximizing the outlive relations.
    // It cannot prove the other direction of the outlive relation.
}
