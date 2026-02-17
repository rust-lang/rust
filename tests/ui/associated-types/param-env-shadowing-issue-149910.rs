// Regression test for issue #149910.
// We want to tell the user about param_env shadowing here.

trait Trait {
    type Assoc;
}

impl<T> Trait for T {
    type Assoc = T;
}

fn foo<T: Trait>(x: T) -> T::Assoc {
    x
    //~^ ERROR mismatched types
}

fn main() {}
