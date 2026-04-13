#![feature(min_generic_const_args, fn_delegation)]
#![allow(incomplete_features)]

mod ice_155125 {
    struct S<const N: usize>;
    impl
        S<
            { //~ ERROR: complex const arguments must be placed inside of a `const` block
                fn foo() {}
                reuse foo; //~ ERROR: the name `foo` is defined multiple times
                2
            },
        >
    {
    }
}

mod ice_155127 {
    struct S;

    fn foo() {}
    impl S {
        #[deprecated] //~ ERROR: `#[deprecated]` attribute cannot be used on delegations
        //~^ WARN: this was previously accepted by the compiler but is being phased out;
        reuse foo;
    }
}

mod ice_155128 {
    fn a() {}

    reuse a as b { //~ ERROR: this function takes 0 arguments but 1 argument was supplied
        fn foo<T>() {};
        foo
    }
}

mod ice_155164 {
    struct X<const N: usize, F> {
        inner: std::iter::Map<
            {
            //~^ ERROR: complex const arguments must be placed inside of a `const` block
            //~| ERROR: constant provided when a type was expected
                struct W<I>;
                impl<I> W<I> {
                    reuse Iterator::fold;
                }
            },
            F,
        >,
    }
}

mod ice_155202 {
    trait Trait {
        fn bar(self);
    }
    impl Trait for () {
        reuse Trait::bar {
            async || {}; //~ ERROR: mismatched types
            //~^ ERROR: cannot find value `async` in this scope
        }
    }
}

fn main() {}
