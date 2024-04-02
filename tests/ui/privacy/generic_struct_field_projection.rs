//! To determine all the types that need to be private when looking at `Struct`, we
//! invoke `predicates_of` to also look at types in `where` bounds.
//! Unfortunately this also computes the inferred outlives bounds, which means for
//! every field we check that if it is of type `&'a T` then `T: 'a` and if it is of
//! struct type, we check that the struct satisfies its lifetime parameters by looking
//! at its inferred outlives bounds. This means we end up with a `<Foo as Trait>::Assoc: 'a`
//! in the outlives bounds of `Struct`. While this is trivially provable, privacy
//! only sees `Foo` and `Trait` and determins that `Foo` is private and then errors.

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
        //~^ ERROR: type `Foo` is private
    }

    pub struct Baz<'a> {
        mode: Bar<'a, Foo>,
    }
}

pub struct Struct<'a> {
    lexer: baz::Baz<'a>,
}

fn main() {}
