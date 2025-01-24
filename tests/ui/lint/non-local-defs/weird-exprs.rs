//@ check-pass
//@ edition:2021

trait Uto {}
struct Test;

type A = [u32; {
    impl Uto for *mut Test {}
    //~^ WARN non-local `impl` definition

    1
}];

enum Enum {
    Discr = {
        impl Uto for Test {}
        //~^ WARN non-local `impl` definition

        1
    }
}

fn main() {
    let _array = [0i32; {
        impl Test {
        //~^ WARN non-local `impl` definition
            fn bar() {}
        }

        1
    }];

    type A = [u32; {
        impl Uto for &Test {}
        //~^ WARN non-local `impl` definition

        1
    }];

    fn a(_: [u32; {
        impl Uto for &(Test,) {}
        //~^ WARN non-local `impl` definition

        1
    }]) {}

    fn b() -> [u32; {
        impl Uto for &(Test,Test) {}
        //~^ WARN non-local `impl` definition

        1
    }] { todo!() }
}
