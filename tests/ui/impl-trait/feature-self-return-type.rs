//@ edition:2018

// This test checks that we emit the correct borrowck error when `Self` or a projection is used as
// a return type.  See #61949 for context.

mod with_self {
    pub struct Foo<'a> {
        pub bar: &'a i32,
    }

    impl<'a> Foo<'a> {
        pub fn new(_bar: &'a i32) -> impl Into<Self> {
            Foo {
                bar: &22
            }
        }
    }

    fn foo() {
        let x = {
            let bar = 22;
            Foo::new(&bar).into()
            //~^ ERROR `bar` does not live long enough
        };
        drop(x);
    }
}

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

        fn make_assoc(self) -> Self::Assoc;
    }

    /// `T::Assoc` can't be normalized any further here.
    fn foo<T: Trait>(x: T) -> impl FooLike<Output = T::Assoc> {
        Foo(x.make_assoc())
    }

    impl<'a> Trait for &'a () {
        type Assoc = &'a ();

        fn make_assoc(self) -> &'a () { &() }
    }

    fn usage() {
        let x = {
            let y = ();
            foo(&y)
            //~^ ERROR `y` does not live long enough
        };
        drop(x);
    }
}

// Same with lifetimes in the trait

mod lifetimes {
    use super::*;

    trait Trait<'a> {
        type Assoc;

        fn make_assoc(self) -> Self::Assoc;
    }

    /// Missing bound constraining `Assoc`, `T::Assoc` can't be normalized further.
    fn foo<'a, T: Trait<'a>>(x: T) -> impl FooLike<Output = T::Assoc> {
        Foo(x.make_assoc())
    }

    impl<'a> Trait<'a> for &'a () {
        type Assoc = &'a ();

        fn make_assoc(self) -> &'a () { &() }
    }

    fn usage() {
        let x = {
            let y = ();
            foo(&y)
            //~^ ERROR `y` does not live long enough
        };
        drop(x);
    }
}

fn main() { }
