//@ check-pass

// Tests that const patterns that use generic parameters are
// allowed if we are still able to evaluate them.

trait Trait { const ASSOC: usize; }

impl<T> Trait for T {
    const ASSOC: usize = 10;
}

fn foo<T>(a: usize) {
    match a {
        <T as Trait>::ASSOC => (),
        _ => (),
    }
}

fn main() {}
