//@ check-pass
//@ edition:2021
//@ aux-build:non_local_macro.rs
//@ rustc-env:CARGO_CRATE_NAME=non_local_def

#![feature(inline_const)]
#![warn(non_local_definitions)]

extern crate non_local_macro;

use std::fmt::{Debug, Display};

struct Test;

impl Debug for Test {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

mod do_not_lint_mod {
    pub trait Tait {}

    impl super::Test {
        fn hugo() {}
    }

    impl Tait for super::Test {}
}

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

    #[macro_export]
    macro_rules! m0 { () => { } };
    //~^ WARN non-local `macro_rules!` definition

    trait Uto4 {}
    impl Uto4 for Test {}

    1
};

trait Uto5 {}
fn main() {
    #[macro_export]
    macro_rules! m { () => { } };
    //~^ WARN non-local `macro_rules!` definition

    impl Test {
    //~^ WARN non-local `impl` definition
        fn foo() {}
    }

    let _array = [0i32; {
        impl Test {
        //~^ WARN non-local `impl` definition
            fn bar() {}
        }

        1
    }];

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

    impl Display for Test {
    //~^ WARN non-local `impl` definition
        fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl dyn Uto5 {}
    //~^ WARN non-local `impl` definition

    impl<T: Uto5> Uto5 for Vec<T> { }
    //~^ WARN non-local `impl` definition

    impl Uto5 for &dyn Uto5 {}
    //~^ WARN non-local `impl` definition

    impl Uto5 for *mut Test {}
    //~^ WARN non-local `impl` definition

    impl Uto5 for *mut [Test] {}
    //~^ WARN non-local `impl` definition

    impl Uto5 for [Test; 8] {}
    //~^ WARN non-local `impl` definition

    impl Uto5 for (Test,) {}
    //~^ WARN non-local `impl` definition

    impl Uto5 for fn(Test) -> () {}
    //~^ WARN non-local `impl` definition

    impl Uto5 for fn() -> Test {}
    //~^ WARN non-local `impl` definition

    let _a = || {
        impl Uto5 for Test {}
        //~^ WARN non-local `impl` definition

        1
    };

    type A = [u32; {
        impl Uto5 for &Test {}
        //~^ WARN non-local `impl` definition

        1
    }];

    fn a(_: [u32; {
        impl Uto5 for &(Test,) {}
        //~^ WARN non-local `impl` definition

        1
    }]) {}

    fn b() -> [u32; {
        impl Uto5 for &(Test,Test) {}
        //~^ WARN non-local `impl` definition

        1
    }] { todo!() }

    struct InsideMain;

    impl Uto5 for *mut InsideMain {}
    //~^ WARN non-local `impl` definition
    impl Uto5 for *mut [InsideMain] {}
    //~^ WARN non-local `impl` definition
    impl Uto5 for [InsideMain; 8] {}
    //~^ WARN non-local `impl` definition
    impl Uto5 for (InsideMain,) {}
    //~^ WARN non-local `impl` definition
    impl Uto5 for fn(InsideMain) -> () {}
    //~^ WARN non-local `impl` definition
    impl Uto5 for fn() -> InsideMain {}
    //~^ WARN non-local `impl` definition

    impl Debug for InsideMain {
        fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl InsideMain {
        fn foo() {}
    }

    fn inside_inside() {
        impl Display for InsideMain {
        //~^ WARN non-local `impl` definition
            fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                todo!()
            }
        }

        impl InsideMain {
        //~^ WARN non-local `impl` definition
            fn bar() {
                #[macro_export]
                macro_rules! m2 { () => { } };
                //~^ WARN non-local `macro_rules!` definition
            }
        }
    }

    trait Uto6 {}
    impl dyn Uto6 {}
    impl Uto5 for dyn Uto6 {}

    impl<T: Uto6> Uto3 for Vec<T> { }
    //~^ WARN non-local `impl` definition
}

trait Uto7 {}
trait Uto8 {}

fn bad() {
    struct Local;
    impl Uto7 for Test where Local: std::any::Any {}
    //~^ WARN non-local `impl` definition

    impl<T> Uto8 for T {}
    //~^ WARN non-local `impl` definition
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

struct UwU<T>(T);

fn fun() {
    #[derive(Debug)]
    struct OwO;
    impl Default for UwU<OwO> {
    //~^ WARN non-local `impl` definition
        fn default() -> Self {
            UwU(OwO)
        }
    }
}

struct Cat;

fn meow() {
    impl From<Cat> for () {
    //~^ WARN non-local `impl` definition
        fn from(_: Cat) -> () {
            todo!()
        }
    }

    #[derive(Debug)]
    struct Cat;
    impl AsRef<Cat> for () {
    //~^ WARN non-local `impl` definition
        fn as_ref(&self) -> &Cat { &Cat }
    }
}

struct G;

fn fun2() {
    #[derive(Debug, Default)]
    struct B;
    impl PartialEq<B> for G {
    //~^ WARN non-local `impl` definition
        fn eq(&self, _: &B) -> bool {
            true
        }
    }
}

fn side_effects() {
    dbg!(().as_ref()); // prints `Cat`
    dbg!(UwU::default().0);
    let _ = G::eq(&G, dbg!(&<_>::default()));
}

struct Dog;

fn woof() {
    impl PartialEq<Dog> for &Dog {
    //~^ WARN non-local `impl` definition
        fn eq(&self, _: &Dog) -> bool {
            todo!()
        }
    }

    impl PartialEq<()> for Dog {
    //~^ WARN non-local `impl` definition
        fn eq(&self, _: &()) -> bool {
            todo!()
        }
    }

    impl PartialEq<()> for &Dog {
    //~^ WARN non-local `impl` definition
        fn eq(&self, _: &()) -> bool {
            todo!()
        }
    }

    impl PartialEq<Dog> for () {
    //~^ WARN non-local `impl` definition
        fn eq(&self, _: &Dog) -> bool {
            todo!()
        }
    }

    struct Test;
    impl PartialEq<Dog> for Test {
        fn eq(&self, _: &Dog) -> bool {
            todo!()
        }
    }
}

struct Wrap<T>(T);

impl Wrap<Wrap<Wrap<()>>> {}

fn rawr() {
    struct Lion;

    impl From<Wrap<Wrap<Lion>>> for () {
    //~^ WARN non-local `impl` definition
        fn from(_: Wrap<Wrap<Lion>>) -> Self {
            todo!()
        }
    }

    impl From<()> for Wrap<Lion> {
    //~^ WARN non-local `impl` definition
        fn from(_: ()) -> Self {
            todo!()
        }
    }
}

macro_rules! m {
    () => {
        trait MacroTrait {}
        struct OutsideStruct;
        fn my_func() {
            impl MacroTrait for OutsideStruct {}
            //~^ WARN non-local `impl` definition
        }
    }
}

m!();

struct CargoUpdate;

non_local_macro::non_local_impl!(CargoUpdate);
//~^ WARN non-local `impl` definition

non_local_macro::non_local_macro_rules!(my_macro);
//~^ WARN non-local `macro_rules!` definition

fn bitflags() {
    struct Flags;

    const _: () = {
        impl Flags {}
    };
}
