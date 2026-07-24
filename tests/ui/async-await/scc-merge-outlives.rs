//@ edition: 2024
//@ revisions: no_dxf dxf
//@[no_dxf] compile-flags: -Znext-solver -Zassumptions-on-binders
//@[dxf] compile-flags: -Znext-solver -Zassumptions-on-binders -Zdxf
//@[dxf] check-pass

#![allow(dead_code)]

use std::marker::PhantomData;

struct Guarded<'a, 'b> {
    _p: PhantomData<(fn(&'a ()) -> &'a (), fn(&'b ()) -> &'b ())>,
    _raw: *mut (),
}

// Send only requires 'b: 'a, not equality.
unsafe impl<'a, 'b: 'a> Send for Guarded<'a, 'b> {}

fn make_guarded<'a>(_x: &mut &'a (), _y: &mut &'a ()) -> Guarded<'a, 'a> {
    Guarded { _p: PhantomData, _raw: std::ptr::null_mut() }
}

async fn use_guarded(data: &()) {
    let mut r1 = data; // '?r1
    let mut r2 = data; // '?r2
    // make_guarded takes &mut &'a (), so 'a is invariant.
    // Both `r1` and `r2` reborrow from `data`, so '?r1 and '?r2 are in the same
    // SCC.
    // This is enough for the Send bound to apply.
    let g = make_guarded(&mut r1, &mut r2);
    std::future::pending::<()>().await;
    drop(g);
}

fn assert_send(_: impl Send) {}

fn main() {
    assert_send(use_guarded(&()));
    //[no_dxf]~^ ERROR `{coroutine witness
}
