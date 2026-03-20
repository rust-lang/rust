// Regression test for #154073.
// Verify that we don't ICE when building vtable entries
// for a trait whose impl is missing a required method body.

//@ compile-flags: --crate-type lib

trait Bar: Sync {
    fn method(&self);
}
impl<T: Sync> Bar for T {
    //~^ ERROR not all trait items implemented
}

static BAR: &dyn Sync = &false as &dyn Bar;
