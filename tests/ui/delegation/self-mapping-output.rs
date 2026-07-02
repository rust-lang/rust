#![feature(fn_delegation)]

mod success {
    trait Trait {
        fn method(&self) -> Self;
        fn r#static() -> Self;
        fn raw_S(&self) -> S { S }
    }

    struct S;
    impl Trait for S {
        fn method(&self) -> S { S }
        fn r#static() -> S { S }
    }

    struct W(S);
    impl Trait for W {
        reuse Trait::method { self.0 }
        reuse Trait::r#static;
        //~^ WARN: function cannot return without recursing [unconditional_recursion]
        reuse Trait::raw_S { self.0 }
    }

    impl W {
        reuse Trait::method { self.0 }
        reuse Trait::r#static;
        reuse Trait::raw_S { self.0 }
    }
}

mod success_non_field {
    trait Trait {
        fn method(&self) -> Self;
        fn r#static() -> Self;
        fn raw_S(&self) -> S { S }
    }

    struct S;
    impl Trait for S {
        fn method(&self) -> S { S }
        fn r#static() -> S { S }
    }

    struct W(S);
    impl Trait for W {
        reuse Trait::method { S }
        reuse Trait::r#static;
        //~^ WARN: function cannot return without recursing [unconditional_recursion]
        reuse Trait::raw_S { S }
    }

    impl W {
        reuse Trait::method { S }
        reuse Trait::r#static;
        reuse Trait::raw_S { S }
    }
}

mod success_generics {
    trait Trait<'a, T, const N: usize> {
        fn method(&self) -> Self;
        fn r#static() -> Self;
        fn raw_S(&self) -> S<'a, 'static, T, (), N, 123> {
            S::<'a, 'static, T, (), N, 123>(std::marker::PhantomData)
        }
    }

    struct S<'a, 'b, A, B, const N: usize, const M: usize>(
        std::marker::PhantomData<&'a &'b (A, B, &'a [(); N], &'b [(); M])>
    );

    impl<'a, T, const N: usize> Trait<'a, T, N> for S<'a, 'static, T, (), N, 123> {
        fn method(&self) -> S<'a, 'static, T, (), N, 123> {
            S(std::marker::PhantomData)
        }

        fn r#static() -> S<'a, 'static, T, (), N, 123> { S(std::marker::PhantomData) }
    }

    struct W<'a, 'b, A, B, const N: usize, const M: usize>(S<'a, 'b, A, B, N, M>);
    impl<'a, T, const N: usize> Trait<'a, T, N> for W<'a, 'static, T, (), N, 123> {
        reuse Trait::<'a, T, N>::method { self.0 }
        reuse Trait::<'a, T, N>::r#static;
        //~^ WARN: function cannot return without recursing [unconditional_recursion]
        reuse Trait::<'a, T, N>::raw_S { self.0 }
    }

    impl<'a, T, const N: usize> W<'a, 'static, T, (), N, 123> {
        reuse Trait::<'a, T, N>::method { self.0 }
        reuse Trait::<'a, T, N>::r#static;
        reuse Trait::<'a, T, N>::raw_S { self.0 }
    }
}

mod no_constructor {
    trait Trait {
        fn method(&self) -> Self;
        fn r#static() -> Self;
        fn raw_S(&self) -> S { S }
    }

    struct S;
    impl Trait for S {
        fn method(&self) -> S { S }
        fn r#static() -> S { S }
    }

    struct W { s: S }
    impl Trait for W {
        reuse Trait::method { self.0 }
        //~^ ERROR: no field `0` on type `&no_constructor::W`
        //~| ERROR: struct `no_constructor::W` has no field named `0`
        reuse Trait::r#static;
        //~^ WARN: function cannot return without recursing [unconditional_recursion]
        reuse Trait::raw_S { self.0 }
        //~^ ERROR: no field `0` on type `&no_constructor::W`
    }

    impl W {
        reuse Trait::method { self.0 }
        //~^ ERROR: no field `0` on type `&no_constructor::W`
        //~| ERROR: struct `no_constructor::W` has no field named `0`
        reuse Trait::r#static;
        reuse Trait::raw_S { self.0 }
        //~^ ERROR: no field `0` on type `&no_constructor::W`
    }
}

mod more_than_one_field {
    trait Trait {
        fn method(&self) -> Self;
        fn r#static() -> Self;
        fn raw_S(&self) -> S { S }
    }

    struct S;
    impl Trait for S {
        fn method(&self) -> S { S }
        fn r#static() -> S { S }
    }

    struct W(S, S, S);
    impl Trait for W {
        reuse Trait::method { self.0 }
        //~^ ERROR: missing fields `1` and `2` in initializer of `more_than_one_field::W`
        reuse Trait::r#static;
        //~^ WARN: function cannot return without recursing [unconditional_recursion]
        reuse Trait::raw_S { self.0 }
    }

    impl W {
        reuse Trait::method { self.0 }
        //~^ ERROR: missing fields `1` and `2` in initializer of `more_than_one_field::W`
        reuse Trait::r#static;
        reuse Trait::raw_S { self.0 }
    }
}

mod non_trait_path_reuse {
    trait Trait {
        fn method(&self) -> Self;
        fn r#static() -> Self;
        fn raw_S(&self) -> S { S }
    }

    mod to_reuse {
        pub fn method(_: impl super::Trait) -> impl super::Trait {
            super::S
        }

        pub fn r#static() -> impl super::Trait {
            super::S
        }
    }

    pub struct S;
    impl Trait for S {
        fn method(&self) -> S { S }
        fn r#static() -> S { S }
    }

    struct W(S);
    impl Trait for W {
        reuse to_reuse::method { self.0 }
        //~^ ERROR: mismatched types
        reuse to_reuse::r#static;
        //~^ ERROR: mismatched types
    }

    impl W {
        reuse to_reuse::method { self.0 }
        //~^ ERROR: no field `0` on type `impl super::Trait`
        reuse to_reuse::r#static;
    }
}

mod non_Self_return_type {
    trait Trait {
        fn method(&self) -> ();
        fn r#static() -> ();
        fn raw_S(&self) -> S { S }
    }

    struct S;
    impl Trait for S {
        fn method(&self) -> () { () }
        fn r#static() -> () { () }
        fn raw_S(&self) -> S { S }
    }

    struct W(());
    impl Trait for W {
        reuse Trait::method { self.0 }
        //~^ ERROR: mismatched types

        reuse Trait::r#static;
        //~^ ERROR: type annotations needed

        reuse Trait::raw_S { self.0 }
        //~^ ERROR: mismatched types
    }

    impl W {
        reuse Trait::method { self.0 }
        //~^ ERROR: mismatched types

        reuse Trait::r#static;
        //~^ ERROR: type annotations needed

        reuse Trait::raw_S { self.0 }
        //~^ ERROR: mismatched types
    }
}

mod wrong_return_type {
    trait Trait {
        fn method(&self) -> Self;
        fn r#static() -> Self;
        fn raw_S(&self) -> S { S }
    }

    struct F;
    impl Trait for F {
        fn method(&self) -> F { F }
        fn r#static() -> F { F }
    }

    struct S;
    impl Trait for S {
        fn method(&self) -> S { S }
        fn r#static() -> S { S }
    }

    struct W(S);
    impl Trait for W {
        reuse <F as Trait>::method { self.0 }
        //~^ ERROR: mismatched types
        //~| ERROR: mismatched types

        reuse <F as Trait>::r#static;
        //~^ ERROR: mismatched types

        reuse <F as Trait>::raw_S { self.0 }
        //~^ ERROR: mismatched types
    }

    impl W {
        reuse <F as Trait>::method { self.0 }
        //~^ ERROR: mismatched types
        //~| ERROR: mismatched types

        reuse <F as Trait>::r#static;
        //~^ ERROR: mismatched types

        reuse <F as Trait>::raw_S { self.0 }
        //~^ ERROR: mismatched types
    }
}

mod wrong_target_expression {
    trait Trait {
        fn method(&self) -> Self;
        fn r#static() -> Self;
        fn raw_S(&self) -> S { S }
    }

    struct S;
    impl Trait for S {
        fn method(&self) -> S { S }
        fn r#static() -> S { S }
    }

    struct F;
    impl Trait for F {
        fn method(&self) -> F { F }
        fn r#static() -> F { F }
    }

    struct W(S);
    impl Trait for W {
        reuse Trait::method { F }
        //~^ ERROR: mismatched types

        reuse Trait::r#static;
        //~^ WARN: function cannot return without recursing [unconditional_recursion]
        reuse Trait::raw_S { F }
    }

    impl W {
        reuse Trait::method { F }
        //~^ ERROR: mismatched types

        reuse Trait::r#static;
        reuse Trait::raw_S { F }
    }
}

mod privacy {
    trait Trait {
        fn method(&self) -> Self;
        fn r#static() -> Self;
        fn raw_S(&self) -> S { S }
    }

    pub struct S;
    impl Trait for S {
        fn method(&self) -> S { S }
        fn r#static() -> S { S }
    }

    mod private {
        pub struct W(super::S);
    }

    impl Trait for private::W {
        reuse Trait::method { self.0 }
        //~^ ERROR: field `0` of struct `private::W` is private

        reuse Trait::r#static;
        //~^ WARN: function cannot return without recursing [unconditional_recursion]

        reuse Trait::raw_S { self.0 }
        //~^ ERROR: field `0` of struct `private::W` is private
    }

    impl private::W {
        reuse Trait::method { self.0 }
        //~^ ERROR: field `0` of struct `private::W` is private

        reuse Trait::r#static;

        reuse Trait::raw_S { self.0 }
        //~^ ERROR: field `0` of struct `private::W` is private
    }
}

fn main() {}
