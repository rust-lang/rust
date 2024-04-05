//! To determine all the types that need to be private when looking at `Struct`, we
//! used to invoke `predicates_of` to also look at types in `where` bounds.
//! Unfortunately this also computes the inferred outlives bounds, which means for
//! every field we check that if it is of type `&'a T` then `T: 'a` and if it is of
//! struct type, we check that the struct satisfies its lifetime parameters by looking
//! at its inferred outlives bounds. This means we end up with a `<Foo as Trait>::Assoc: 'a`
//! in the outlives bounds of `Struct`. While this is trivially provable, privacy
//! only sees `Foo` and `Trait` and determines that `Foo` is private and then errors.
//! So now we invoke `explicit_predicates_of` to make sure we only care about user-written
//! predicates.

//@ check-pass

mod baz {
    struct Foo;

    pub trait Trait {
        type Assoc;
    }

    impl Trait for Foo {
        type Assoc = ();
    }

    pub struct Bar<'a, T: Trait> {
        source: &'a T::Assoc,
    }

    pub struct Baz<'a> {
        mode: Bar<'a, Foo>,
    }
}

pub struct Struct<'a> {
    lexer: baz::Baz<'a>,
}

fn main() {}
