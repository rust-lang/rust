//! Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/219>.
//@compile-flags: -Znext-solver=globally
//@ check-pass

trait Trait {
    type Assoc<V>;
    fn foo<V>()
    where
        Self::Assoc<V>: Trait;
}

impl<T> Trait for T {
    type Assoc<V> = T;
    fn foo<V>() {}
}

fn main() {}
