#![warn(clippy::unsafe_derive_deserialize)]
#![allow(unused, clippy::missing_safety_doc)]

extern crate serde;

use serde::Deserialize;

#[derive(Deserialize)]
//~^ ERROR: you are deriving `serde::Deserialize` on a type that has methods using `unsafe
pub struct A;
impl A {
    pub unsafe fn new(_a: i32, _b: i32) -> Self {
        Self {}
    }
}

#[derive(Deserialize)]
//~^ ERROR: you are deriving `serde::Deserialize` on a type that has methods using `unsafe
pub struct B;
impl B {
    pub unsafe fn unsafe_method(&self) {}
}

#[derive(Deserialize)]
//~^ ERROR: you are deriving `serde::Deserialize` on a type that has methods using `unsafe
pub struct C;
impl C {
    pub fn unsafe_block(&self) {
        unsafe {}
    }
}

#[derive(Deserialize)]
//~^ ERROR: you are deriving `serde::Deserialize` on a type that has methods using `unsafe
pub struct D;
impl D {
    pub fn inner_unsafe_fn(&self) {
        unsafe fn inner() {}
    }
}

// Does not derive `Deserialize`, should be ignored
pub struct E;
impl E {
    pub unsafe fn new(_a: i32, _b: i32) -> Self {
        Self {}
    }

    pub unsafe fn unsafe_method(&self) {}

    pub fn unsafe_block(&self) {
        unsafe {}
    }

    pub fn inner_unsafe_fn(&self) {
        unsafe fn inner() {}
    }
}

// Does not have methods using `unsafe`, should be ignored
#[derive(Deserialize)]
pub struct F;

// Check that we honor the `allow` attribute on the ADT
#[allow(clippy::unsafe_derive_deserialize)]
#[derive(Deserialize)]
pub struct G;
impl G {
    pub fn unsafe_block(&self) {
        unsafe {}
    }
}

// Check that we honor the `expect` attribute on the ADT
#[expect(clippy::unsafe_derive_deserialize)]
#[derive(Deserialize)]
pub struct H;
impl H {
    pub fn unsafe_block(&self) {
        unsafe {}
    }
}

fn main() {}

mod issue15120 {
    macro_rules! uns {
        ($e:expr) => {
            unsafe { $e }
        };
    }

    #[derive(serde::Deserialize)]
    struct Foo;

    impl Foo {
        fn foo(&self) {
            // Do not lint if `unsafe` comes from the `core::pin::pin!()` macro
            std::pin::pin!(());
        }
    }

    //~v unsafe_derive_deserialize
    #[derive(serde::Deserialize)]
    struct Bar;

    impl Bar {
        fn bar(&self) {
            // Lint if `unsafe` comes from the another macro
            _ = uns!(42);
        }
    }
}
