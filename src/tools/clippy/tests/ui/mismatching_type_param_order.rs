#![warn(clippy::mismatching_type_param_order)]
#![allow(clippy::disallowed_names)]

fn main() {
    struct Foo<A, B> {
        x: A,
        y: B,
    }

    // lint on both params
    impl<B, A> Foo<B, A> {}

    // lint on the 2nd param
    impl<C, A> Foo<C, A> {}

    // should not lint
    impl<A, B> Foo<A, B> {}

    struct FooLifetime<'l, 'm, A, B> {
        x: &'l A,
        y: &'m B,
    }

    // should not lint on lifetimes
    impl<'m, 'l, B, A> FooLifetime<'m, 'l, B, A> {}

    struct Bar {
        x: i32,
    }

    // should not lint
    impl Bar {}

    // also works for enums
    enum FooEnum<A, B, C> {
        X(A),
        Y(B),
        Z(C),
    }

    impl<C, A, B> FooEnum<C, A, B> {}

    // also works for unions
    union FooUnion<A: Copy, B>
    where
        B: Copy,
    {
        x: A,
        y: B,
    }

    impl<B: Copy, A> FooUnion<B, A> where A: Copy {}

    impl<A, B> FooUnion<A, B>
    where
        A: Copy,
        B: Copy,
    {
    }

    // if the types are complicated, do not lint
    impl<K, V, B> Foo<(K, V), B> {}
    impl<K, V, A> Foo<(K, V), A> {}
}
