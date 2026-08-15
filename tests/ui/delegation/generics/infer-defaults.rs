#![feature(fn_delegation)]

mod free_to_trait {
    trait Trait<'a, XX, Y, T = (), const N: usize = 2> {
        fn foo<A, B>(&self, t: (T, A, B), slice: &[usize; N]) {}
    }

    struct X;
    impl<XX, Y, T, const N: usize> Trait<'_, XX, Y, T, N> for X {}

    // When infer is specified for default parameter the generic param is generated.
    reuse Trait::<'_, _, _, _, _>::foo as foo;
    // When default params are omitted they are not generated but used in signature inheritance.
    reuse Trait::<'_, _, _>::foo as bar;

    // Check with user specified args in child:
    // When infer is specified for default parameter the generic param is generated.
    reuse Trait::<'_, _, _, _, _>::foo::<(), _> as foo1;
    // When default params are omitted they are not generated but used in signature inheritance.
    reuse Trait::<'_, _, _>::foo::<_, ()> as bar1;

    // Check with explicit self type.
    reuse <X as Trait::<'_, _, _, _, _>>::foo as foo2;
    reuse <X as Trait::<'_, _, _>>::foo as bar2;

    reuse <X as Trait::<'_, _, _, _, _>>::foo::<(), _> as foo3;
    reuse <X as Trait::<'_, _, _>>::foo::<_, ()> as bar3;

    fn check() {
        foo::<'static, X, (), (), (), 1, (), ()>(&X, ((), (), ()), &[1]);
        bar::<'static, X, (), (), (), ()>(&X, ((), (), ()), &[1, 2]);

        foo1::<'static, X, (), (), (), 2, ()>(&X, ((), (), ()), &[1, 2]);
        bar1::<'static, X, (), (), ()>(&X, ((), (), ()), &[1, 2]);

        foo2::<'static, (), (), (), 3, (), ()>(&X, ((), (), ()), &[1, 2, 3]);
        bar2::<'static, (), (), (), ()>(&X, ((), (), ()), &[1, 2]);

        foo3::<'static, (), (), (), 4, ()>(&X, ((), (), ()), &[1, 2, 3, 4]);
        bar3::<'static, (), (), ()>(&X, ((), (), ()), &[1, 2]);
    }
}

mod trait_to_trait {
    trait Trait<'a, X, Y, T = (), const N: usize = 2> {
        fn foo(&self, t: (T, T, T), slice: &[usize; N]) {}
        fn bar(&self, t: (T, T, T), slice: &[usize; N]) {}
    }

    trait Trait2<'a, X, Y>: Trait<'a, X, Y> {
        // Default params are generated as usual generics as infers are specified.
        reuse Trait::<'a, X, Y, _, _>::foo;
        //~^ ERROR: the trait bound `Self: trait_to_trait::Trait<'a, X, Y, T, N>` is not satisfied

        // Default params are not generated, as they are not specified.
        reuse Trait::<'a, X, Y>::foo as bar;
    }

    impl Trait<'static, (), ()> for () {}
    impl Trait2<'static, (), ()> for () {}

    fn check() {
        Trait2::<'static, (), ()>::foo::<(), 1>(&(), ((), (), ()), &[1]);
        Trait2::<'static, (), ()>::bar(&(), ((), (), ()), &[1, 2]);
    }
}

mod trait_impl_to_trait {
    trait Trait<'a, X, Y, T = (), const N: usize = 2> {
        fn foo(&self, t: (T, T, T), slice: &[usize; N]) {}
        fn bar(&self, t: (T, T, T), slice: &[usize; N]) {}
    }

    struct S;
    impl<X, Y> Trait<'_, X, Y> for S {}

    struct W(S);
    impl<X, Y> Trait<'_, X, Y> for W {
        // Generics of both methods match generics of their signature
        // functions in `Trait` declaration, no matter specified infers.
        reuse Trait::<'static, X, Y>::foo { self.0 }
        reuse Trait::<'static, X, Y, _, _>::foo as bar { self.0 }
    }

    fn check() {
        W(S).foo(((), (), ()), &[1, 2]);
        W(S).foo::<1, 2, 3>(((), (), ()), &[1, 2]);
        //~^ ERROR: method takes 0 generic arguments but 3 generic arguments were supplied

        W(S).bar(((), (), ()), &[1]);
        //~^ ERROR: mismatched types
        W(S).bar::<((), ()), 0>(((), (), ()), &[1, 2]);
        //~^ ERROR: method takes 0 generic arguments but 2 generic arguments were supplied

        W(S).bar(((), (), ()), &[1, 2]);
    }
}

mod inherent_impl_to_trait {
    trait Trait<'a, X, Y, T = (), const N: usize = 2> {
        fn foo(&self, t: (T, T, T), slice: &[usize; N]) {}
    }

    struct S<T>(T);

    impl<T: Trait<'static, (), ()>> S<T> {
        // Default params are not generated, as they are not specified.
        reuse Trait::<'static, (), ()>::foo { self.0 }

        // Default params are generated as usual generics as infers are specified.
        reuse Trait::<'static, (), (), _, _>::foo as bar { self.0 }
        //~^ ERROR: the trait bound `T: inherent_impl_to_trait::Trait<'static, (), (), T, N>` is not satisfied
    }

    impl Trait<'static, (), ()> for () {}

    fn check() {
        S(()).foo(((), (), ()), &[1, 2]);
        S(()).bar(((), (), ()), &[1, 2]);
        S(()).bar::<usize, 4>((1, 2, 3), &[1, 2, 3, 4]);
    }
}

fn main() {}
