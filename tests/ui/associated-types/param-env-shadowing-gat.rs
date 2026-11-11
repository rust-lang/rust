// Regression test for issue #149910.
// This ensures that the diagnostics logic handles Generic Associated Types (GATs)
// correctly without crashing (ICE).

trait Trait {
    type Assoc<T>;
}

impl<T> Trait for T {
    type Assoc<U> = U;
}

fn foo<T: Trait>(x: T::Assoc<T>) -> u32 {
    x //~ ERROR mismatched types
}

fn main() {}
