//@ edition:2018

// See issue 60414

// Reduction to `impl Trait`

struct Foo<T>(T);

trait FooLike {
    type Output;
}

impl<T> FooLike for Foo<T> {
    type Output = T;
}

mod impl_trait {
    use super::*;

    trait Trait {
        type Assoc;
    }

    /// `T::Assoc` can't be normalized any further here.
    fn foo_fail<T: Trait>() -> impl FooLike<Output = T::Assoc> {
        //~^ ERROR: type mismatch
        Foo(())
    }
}

// Same with lifetimes in the trait

mod lifetimes {
    use super::*;

    trait Trait<'a> {
        type Assoc;
    }

    /// Missing bound constraining `Assoc`, `T::Assoc` can't be normalized further.
    fn foo2_fail<'a, T: Trait<'a>>() -> impl FooLike<Output = T::Assoc> {
        //~^ ERROR: type mismatch
        Foo(())
    }
}

fn main() {}
