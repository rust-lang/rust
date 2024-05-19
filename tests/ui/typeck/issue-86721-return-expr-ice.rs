// Regression test for the ICE described in #86721.

//@ revisions: rev1 rev2
#![cfg_attr(any(), rev1, rev2)]
#![crate_type = "lib"]

#[cfg(any(rev1))]
trait T {
    const U: usize = return;
    //[rev1]~^ ERROR: return statement outside of function body [E0572]
}

#[cfg(any(rev2))]
trait T2 {
    fn foo(a: [(); return]);
    //[rev2]~^ ERROR: return statement outside of function body [E0572]
}
