// check-pass
// edition:2018
// revisions: default sa

#![feature(type_alias_impl_trait)]

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

    /// `T::Assoc` should be normalized to `()` here.
    fn foo_pass<T: Trait<Assoc = ()>>() -> impl FooLike<Output = T::Assoc> {
        Foo(())
    }
}

// Same with lifetimes in the trait

mod lifetimes {
    use super::*;

    trait Trait<'a> {
        type Assoc;
    }

    /// Like above.
    ///
    /// FIXME(#51525) -- the shorter notation `T::Assoc` winds up referencing `'static` here
    fn foo2_pass<'a, T: Trait<'a, Assoc = ()> + 'a>()
    -> impl FooLike<Output = <T as Trait<'a>>::Assoc> + 'a {
        Foo(())
    }

    /// Normalization to type containing bound region.
    ///
    /// FIXME(#51525) -- the shorter notation `T::Assoc` winds up referencing `'static` here
    fn foo2_pass2<'a, T: Trait<'a, Assoc = &'a ()> + 'a>()
    -> impl FooLike<Output = <T as Trait<'a>>::Assoc> + 'a {
        Foo(&())
    }
}

// The same applied to `type Foo = impl Bar`s

mod opaque_types {
    trait Implemented {
        type Assoc;
    }
    impl<T> Implemented for T {
        type Assoc = u8;
    }

    trait Trait {
        type Out;
    }

    impl Trait for () {
        type Out = u8;
    }

    type Ex = impl Trait<Out = <() as Implemented>::Assoc>;

    fn define() -> Ex {
        ()
    }
}

fn main() {}
