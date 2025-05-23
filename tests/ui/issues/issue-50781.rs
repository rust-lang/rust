trait Trait {}

trait X {
    fn foo(&self) where Self: Trait;
}

impl X for () {
    fn foo(&self) {}
}

impl Trait for dyn X {}
//~^ ERROR the trait `X` is not dyn compatible

pub fn main() {
    // Check that this does not segfault.
    <dyn X as X>::foo(&());
    //~^ ERROR the trait `X` is not dyn compatible
}
