#![deny(where_clauses_object_safety)]

trait Trait {}

trait X {
    fn foo(&self) where Self: Trait; //~ ERROR the trait `X` cannot be made into an object
    //~^ WARN this was previously accepted by the compiler but is being phased out
}

impl X for () {
    fn foo(&self) {}
}

impl Trait for dyn X {}

pub fn main() {
    // Check that this does not segfault.
    <dyn X as X>::foo(&());
}
