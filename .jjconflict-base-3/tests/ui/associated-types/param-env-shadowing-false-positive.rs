// Regression test for issue #149910.
// The compiler previously incorrectly claimed that the local param-env bound
// shadowed the global impl, but they are actually the same.

trait Trait {
    type Assoc;
}

impl<T> Trait for T {
    type Assoc = T;
}

fn foo<T: Trait>(x: T::Assoc) -> u32 {
    x //~ ERROR mismatched types
}

fn main() {}
