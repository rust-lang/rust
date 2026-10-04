//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:generics.pp

#![allow(incomplete_features)]
#![attr = Feature([fn_delegation#0])]
extern crate std;
#[attr = PreludeImport]
use ::std::prelude::rust_2015::*;

mod free_to_trait {
    trait Trait<'a, XX, Y, T = (), const N: usize = 2> {
        fn method<A, B>(&self, t: (T, A, B), slice: &'_ [usize; N]) { }
        fn r#static<A, B>(t: (T, A, B), slice: &'_ [usize; N]) { }
    }

    struct X;
    impl <XX, Y, T, const N: usize> Trait<'_, XX, Y, T, N> for X { }

    // When infer is specified for default parameter the generic param is generated.
    #[attr = Inline(Hint)]
    fn foo<'a, Self, XX, Y, T, const N: _, A, B>(self: _, arg1: _, arg2: _)
        -> _ where
        'a:'a {
        <Self as Trait::<'a, XX, Y, T, N>>::method::<A, B>(self, arg1, arg2)
    }
    #[attr = Inline(Hint)]
    fn static_foo<'a, Self, XX, Y, T, const N: _, A, B>(arg0: _, arg1: _) -> _
        where
        'a:'a {
        <Self as Trait::<'a, XX, Y, T, N>>::r#static::<A, B>(arg0, arg1)
    }

    // When default params are omitted they are not generated but used in signature inheritance.
    #[attr = Inline(Hint)]
    fn bar<'a, Self, XX, Y, A, B>(self: _, arg1: _, arg2: _) -> _ where
        'a:'a {
        <Self as Trait::<'a, XX, Y>>::method::<A, B>(self, arg1, arg2)
    }
    #[attr = Inline(Hint)]
    fn static_bar<'a, Self, XX, Y, A, B>(arg0: _, arg1: _) -> _ where
        'a:'a { <Self as Trait::<'a, XX, Y>>::r#static::<A, B>(arg0, arg1) }

    // Check with user specified args in child:
    // When infer is specified for default parameter the generic param is generated.
    #[attr = Inline(Hint)]
    fn foo1<'a, Self, XX, Y, T, const N: _, B>(self: _, arg1: _, arg2: _) -> _
        where
        'a:'a {
        <Self as Trait::<'a, XX, Y, T, N>>::method::<(), B>(self, arg1, arg2)
    }
    #[attr = Inline(Hint)]
    fn static_foo1<'a, Self, XX, Y, T, const N: _, B>(arg0: _, arg1: _) -> _
        where
        'a:'a {
        <Self as Trait::<'a, XX, Y, T, N>>::r#static::<(), B>(arg0, arg1)
    }

    // When default params are omitted they are not generated but used in signature inheritance.
    #[attr = Inline(Hint)]
    fn bar1<'a, Self, XX, Y, A>(self: _, arg1: _, arg2: _) -> _ where
        'a:'a {
        <Self as Trait::<'a, XX, Y>>::method::<A, ()>(self, arg1, arg2)
    }
    #[attr = Inline(Hint)]
    fn static_bar1<'a, Self, XX, Y, A>(arg0: _, arg1: _) -> _ where
        'a:'a { <Self as Trait::<'a, XX, Y>>::r#static::<A, ()>(arg0, arg1) }

    // Check with explicit self type.
    #[attr = Inline(Hint)]
    fn foo2<'a, XX, Y, T, const N: _, A, B>(self: _, arg1: _, arg2: _) -> _
        where
        'a:'a {
        <X as Trait::<'a, XX, Y, T, N>>::method::<A, B>(self, arg1, arg2)
    }
    #[attr = Inline(Hint)]
    fn static_foo2<'a, XX, Y, T, const N: _, A, B>(arg0: _, arg1: _) -> _
        where
        'a:'a {
        <X as Trait::<'a, XX, Y, T, N>>::r#static::<A, B>(arg0, arg1)
    }

    #[attr = Inline(Hint)]
    fn bar2<'a, XX, Y, A, B>(self: _, arg1: _, arg2: _) -> _ where
        'a:'a { <X as Trait::<'a, XX, Y>>::method::<A, B>(self, arg1, arg2) }
    #[attr = Inline(Hint)]
    fn static_bar2<'a, XX, Y, A, B>(arg0: _, arg1: _) -> _ where
        'a:'a { <X as Trait::<'a, XX, Y>>::r#static::<A, B>(arg0, arg1) }

    #[attr = Inline(Hint)]
    fn foo3<'a, XX, Y, T, const N: _, B>(self: _, arg1: _, arg2: _) -> _ where
        'a:'a {
        <X as Trait::<'a, XX, Y, T, N>>::method::<(), B>(self, arg1, arg2)
    }
    #[attr = Inline(Hint)]
    fn static_foo3<'a, XX, Y, T, const N: _, B>(arg0: _, arg1: _) -> _ where
        'a:'a {
        <X as Trait::<'a, XX, Y, T, N>>::r#static::<(), B>(arg0, arg1)
    }

    #[attr = Inline(Hint)]
    fn bar3<'a, XX, Y, A>(self: _, arg1: _, arg2: _) -> _ where
        'a:'a { <X as Trait::<'a, XX, Y>>::method::<A, ()>(self, arg1, arg2) }
    #[attr = Inline(Hint)]
    fn static_bar3<'a, XX, Y, A>(arg0: _, arg1: _) -> _ where
        'a:'a { <X as Trait::<'a, XX, Y>>::r#static::<A, ()>(arg0, arg1) }
}

mod trait_impl_to_trait {
    trait Trait<'a, X, Y, T = (), const N: usize = 2> {
        fn foo(&self, t: (T, T, T), slice: &'_ [usize; N]) { }
        fn bar(&self, t: (T, T, T), slice: &'_ [usize; N]) { }
    }

    struct S;
    impl <X, Y> Trait<'_, X, Y> for S { }

    struct W(S);
    impl <X, Y> Trait<'_, X, Y> for W {
        // Generics of both methods match generics of their signature
        // functions in `Trait` declaration, no matter specified infers.
        #[attr = Inline(Hint)]
        fn foo(self: _, arg1: _, arg2: _)
            -> _ { Trait::<'static, X, Y>::foo(self.0, arg1, arg2) }
        #[attr = Inline(Hint)]
        fn bar(self: _, arg1: _, arg2: _)
            -> _ { Trait::<'static, X, Y, _, _>::foo(self.0, arg1, arg2) }
    }
}

fn main() { }
