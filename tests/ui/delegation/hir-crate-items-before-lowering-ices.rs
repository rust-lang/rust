//@ revisions: ice_155125 ice_155127 ice_155128 ice_155164 ice_155202
//@[ice_155202] edition: 2018..

#![feature(min_generic_const_args, fn_delegation)]

#[cfg(ice_155125)]
mod ice_155125 {
    struct S<const N: usize>;
    impl
        S<
            core::direct_const_arg!({ //[ice_155125]~ ERROR: complex const arguments must be placed inside of a `const` block
                fn foo() {}
                reuse foo; //[ice_155125]~ ERROR: the name `foo` is defined multiple times
                2
            }),
        >
    {
    }
}

#[cfg(ice_155127)]
mod ice_155127 {
    struct S;

    fn foo() {}
    impl S {
        #[deprecated] //[ice_155127]~ ERROR: the `deprecated` attribute cannot be used on delegations
        //[ice_155127]~^ WARN: this was previously accepted by the compiler but is being phased out;
        reuse foo;
    }
}

#[cfg(ice_155128)]
mod ice_155128 {
    fn a() {}

    reuse a as b { //[ice_155128]~ ERROR: delegation's target expression is specified for function with no params
        //[ice_155128]~^ ERROR: this function takes 0 arguments but 1 argument was supplied
        fn foo<T>() {};
        foo
    }
}

#[cfg(ice_155164)]
mod ice_155164 {
    struct X<const N: usize, F> {
        inner: std::iter::Map<
            core::direct_const_arg!({
            //[ice_155164]~^ ERROR: complex const arguments must be placed inside of a `const` block
                struct W<I>;
                impl<I> W<I> {
                    reuse Iterator::fold;
                }
            }),
            F,
        >,
    }
}

#[cfg(ice_155202)]
#[deny(unconditional_recursion)]
mod ice_155202 {
    trait Trait {
        fn bar(self);
    }
    impl Trait for () {
        reuse Trait::bar { //[ice_155202]~ ERROR: cannot return without recursing
            async || {};
        }
    }
}

fn main() {}
