// Regression test for the ICE described in #86721.

#![crate_type="lib"]

trait T {
    const U: usize = return;
    //~^ ERROR: return statement outside of function body [E0572]
}
