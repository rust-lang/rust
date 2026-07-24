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

// Send requires 'a: 'b outlives instead of equality.
unsafe impl<'a: 'b, 'b> Send for Guarded<'a, 'b> {}

fn make_guarded<'a, 'b>(_x: &mut &'a (), _y: &mut &'b ()) -> Guarded<'a, 'b> {
    Guarded { _p: PhantomData, _raw: std::ptr::null_mut() }
}

async fn use_guarded(data: &()) {
    let mut r1 = data;
    let mut r2 = data;
    let g = make_guarded(&mut r1, &mut r2);
    std::future::pending::<()>().await;
    drop(g);
}

fn assert_send(_: impl Send) {}

fn main() {
    assert_send(use_guarded(&()));
    //[no_dxf]~^ ERROR `{coroutine witness
}
