//@ edition: 2024
//@ compile-flags: -Znext-solver -Zassumptions-on-binders -Zdxf

#![allow(dead_code)]

// Under -Zdxf, the lifetime error should be stalled/ambiguous, but the Rc error
// should fail faster.
// We should only see the Rc error.

use std::marker::PhantomData;
use std::rc::Rc;
 
struct Guarded<'a, 'b> {
    _p: PhantomData<(fn(&'a ()) -> &'a (), fn(&'b ()) -> &'b ())>,
    _raw: *mut (),
}
 
// Send requires 'b: 'a (outlives, not just equality).
unsafe impl<'a, 'b: 'a> Send for Guarded<'a, 'b> {}
 
fn make_guarded<'a>(_x: &mut &'a (), _y: &mut &'a ()) -> Guarded<'a, 'a> {
    Guarded { _p: PhantomData, _raw: std::ptr::null_mut() }
}
 
async fn use_guarded_and_rc(data: &()) {
    let mut r1 = data;
    let mut r2 = data;
    let g = make_guarded(&mut r1, &mut r2);
    let rc = Rc::new(42);
    std::future::pending::<()>().await;
    drop(g);
    drop(rc);
}
 
fn assert_send(_: impl Send) {}
 
fn main() {
    assert_send(use_guarded_and_rc(&()));
    //~^ ERROR future cannot be sent between threads safely
    // We should NOT see an error about `Guarded` or coroutine witness Send here,
    // as that part is stalled.
}
