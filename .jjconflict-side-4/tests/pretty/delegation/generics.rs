//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:generics.pp

#![allow(incomplete_features)]
#![feature(fn_delegation)]

mod free_to_trait {
    trait Trait<'a, XX, Y, T = (), const N: usize = 2> {
        fn method<A, B>(&self, t: (T, A, B), slice: &[usize; N]) {}
        fn r#static<A, B>(t: (T, A, B), slice: &[usize; N]) {}
    }

    struct X;
    impl<XX, Y, T, const N: usize> Trait<'_, XX, Y, T, N> for X {}

    // When infer is specified for default parameter the generic param is generated.
    reuse Trait::<'_, _, _, _, _>::method as foo;
    reuse Trait::<'_, _, _, _, _>::r#static as static_foo;

    // When default params are omitted they are not generated but used in signature inheritance.
    reuse Trait::<'_, _, _>::method as bar;
    reuse Trait::<'_, _, _>::r#static as static_bar;

    // Check with user specified args in child:
    // When infer is specified for default parameter the generic param is generated.
    reuse Trait::<'_, _, _, _, _>::method::<(), _> as foo1;
    reuse Trait::<'_, _, _, _, _>::r#static::<(), _> as static_foo1;

    // When default params are omitted they are not generated but used in signature inheritance.
    reuse Trait::<'_, _, _>::method::<_, ()> as bar1;
    reuse Trait::<'_, _, _>::r#static::<_, ()> as static_bar1;

    // Check with explicit self type.
    reuse <X as Trait::<'_, _, _, _, _>>::method as foo2;
    reuse <X as Trait::<'_, _, _, _, _>>::r#static as static_foo2;

    reuse <X as Trait::<'_, _, _>>::method as bar2;
    reuse <X as Trait::<'_, _, _>>::r#static as static_bar2;

    reuse <X as Trait::<'_, _, _, _, _>>::method::<(), _> as foo3;
    reuse <X as Trait::<'_, _, _, _, _>>::r#static::<(), _> as static_foo3;

    reuse <X as Trait::<'_, _, _>>::method::<_, ()> as bar3;
    reuse <X as Trait::<'_, _, _>>::r#static::<_, ()> as static_bar3;
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
}

fn main() {}
