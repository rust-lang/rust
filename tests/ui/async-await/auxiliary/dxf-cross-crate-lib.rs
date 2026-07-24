//@ edition: 2024
//@ compile-flags: -Znext-solver -Zassumptions-on-binders -Zdxf

#![allow(dead_code)]

use std::marker::PhantomData;

pub struct Guarded<'a, 'b> {
    _p: PhantomData<(fn(&'a ()) -> &'a (), fn(&'b ()) -> &'b ())>,
    _raw: *mut (),
}

unsafe impl<'a, 'b: 'a> Send for Guarded<'a, 'b> {}

pub fn make_guarded<'a>(_x: &mut &'a (), _y: &mut &'a ()) -> Guarded<'a, 'a> {
    Guarded { _p: PhantomData, _raw: std::ptr::null_mut() }
}

pub async fn use_guarded(data: &()) {
    let mut r1 = data;
    let mut r2 = data;
    let g = make_guarded(&mut r1, &mut r2);
    std::future::pending::<()>().await;
    drop(g);
}
