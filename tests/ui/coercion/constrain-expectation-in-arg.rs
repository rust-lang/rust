//@ check-pass

// Regression test for for #129286.
// Makes sure that we don't have unconstrained type variables that come from
// bivariant type parameters due to the way that we construct expectation types
// when checking call expressions in HIR typeck.

trait Trait {
    type Item;
}

struct Struct<A: Trait<Item = B>, B> {
    pub field: A,
}

fn identity<T>(x: T) -> T {
    x
}

fn test<A: Trait<Item = B>, B>(x: &Struct<A, B>) {
    let x: &Struct<_, _> = identity(x);
}

fn main() {}
