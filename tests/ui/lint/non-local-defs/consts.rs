//@ check-pass
//@ edition:2021
//@ rustc-env:CARGO_CRATE_NAME=non_local_def

struct Test;

trait Uto {}
const Z: () = {
    trait Uto1 {}

    impl Uto1 for Test {} // the trait is local, don't lint

    impl Uto for &Test {}
    //~^ WARN non-local `impl` definition
};

trait Ano {}
const _: () = {
    impl Ano for &Test {} // ignored since the parent is an anon-const
};

trait Uto2 {}
static A: u32 = {
    impl Uto2 for Test {}
    //~^ WARN non-local `impl` definition

    1
};

trait Uto3 {}
const B: u32 = {
    impl Uto3 for Test {}
    //~^ WARN non-local `impl` definition

    trait Uto4 {}
    impl Uto4 for Test {}

    1
};

trait Uto5 {}
fn main() {
    impl Test {
    //~^ WARN non-local `impl` definition
        fn foo() {}
    }


    const {
        impl Test {
        //~^ WARN non-local `impl` definition
            fn hoo() {}
        }

        1
    };

    const _: u32 = {
        impl Test {
        //~^ WARN non-local `impl` definition
            fn foo2() {}
        }

        1
    };

    const _: () = {
        const _: () = {
            impl Test {}
            //~^ WARN non-local `impl` definition
        };
    };
}

trait Uto9 {}
trait Uto10 {}
const _: u32 = {
    let _a = || {
        impl Uto9 for Test {}
        //~^ WARN non-local `impl` definition

        1
    };

    type A = [u32; {
        impl Uto10 for Test {}
        //~^ WARN non-local `impl` definition

        1
    }];

    1
};
