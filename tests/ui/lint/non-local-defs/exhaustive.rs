//@ check-pass
//@ edition:2021

use std::fmt::Display;

trait Trait {}
struct Test;

fn main() {
    impl Test {
    //~^ WARN non-local `impl` definition
        fn foo() {}
    }

    impl Display for Test {
    //~^ WARN non-local `impl` definition
        fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl dyn Trait {}
    //~^ WARN non-local `impl` definition

    impl<T: Trait> Trait for Vec<T> { }
    //~^ WARN non-local `impl` definition

    impl Trait for &dyn Trait {}
    //~^ WARN non-local `impl` definition

    impl Trait for *mut Test {}
    //~^ WARN non-local `impl` definition

    impl Trait for *mut [Test] {}
    //~^ WARN non-local `impl` definition

    impl Trait for [Test; 8] {}
    //~^ WARN non-local `impl` definition

    impl Trait for (Test,) {}
    //~^ WARN non-local `impl` definition

    impl Trait for fn(Test) -> () {}
    //~^ WARN non-local `impl` definition

    impl Trait for fn() -> Test {}
    //~^ WARN non-local `impl` definition

    let _a = || {
        impl Trait for Test {}
        //~^ WARN non-local `impl` definition

        1
    };

    struct InsideMain;

    impl Trait for &InsideMain {}
    impl Trait for *mut InsideMain {}
    impl Trait for *mut [InsideMain] {}
    impl Trait for [InsideMain; 8] {}
    impl Trait for (InsideMain,) {}
    impl Trait for fn(InsideMain) -> () {}
    impl Trait for fn() -> InsideMain {}

    fn inside_inside() {
        impl Display for InsideMain {
        //~^ WARN non-local `impl` definition
            fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                todo!()
            }
        }

        impl InsideMain {
        //~^ WARN non-local `impl` definition
            fn bar() {}
        }
    }
}
