//@ revisions: sdylib binary
// Currently the two revisions have different expectations, this is wrong.
// There is an issue open to fix this: https://github.com/rust-lang/rust/issues/158227

//@ compile-flags: -Zunstable-options -Csymbol-mangling-version=v0
//@[sdylib] compile-flags: --crate-type=sdylib
//@[binary] compile-flags: --crate-type=bin

#![allow(incomplete_features, improper_ctypes_definitions)]
#![feature(export_stable)]
#![feature(inherent_associated_types)]

mod m {
    #[export_stable]
    pub struct S;
    //[sdylib]~^ ERROR private items are not exportable

    pub fn foo() -> i32 { 0 }
    //[sdylib]~^ ERROR only functions with "C" ABI are exportable
}

#[export_stable]
pub use m::foo;

#[export_stable]
pub mod m1 {
    #[repr(C)]
    pub struct S1; // OK, public type with stable repr

    struct S2;

    pub struct S3;
    //[sdylib]~^ ERROR types with unstable layout are not exportable
}

pub mod fn_sig {
    #[export_stable]
    pub fn foo1() {}
    //[sdylib]~^ ERROR only functions with "C" ABI are exportable

    #[export_stable]
    #[repr(C)]
    pub struct S;

    #[export_stable]
    pub extern "C" fn foo2(x: S) -> i32 { 0 }

    #[export_stable]
    pub extern "C" fn foo3(x: Box<S>) -> i32 { 0 }
    //[sdylib]~^ ERROR function with `#[export_stable]` attribute uses type `Box<fn_sig::S>`, which is not exportable
}

pub mod impl_item {
    pub struct S;

    impl S {
        #[export_stable]
        pub extern "C" fn foo1(&self) -> i32 { 0 }
        //[sdylib]~^ ERROR method with `#[export_stable]` attribute uses type `&impl_item::S`, which is not exportable

        #[export_stable]
        pub extern "C" fn foo2(self) -> i32 { 0 }
        //[sdylib]~^ ERROR method with `#[export_stable]` attribute uses type `impl_item::S`, which is not exportable
    }

    pub struct S2<T>(T);

    impl<T> S2<T> {
        #[export_stable]
        pub extern "C" fn foo1(&self) {}
        //[sdylib]~^ ERROR generic functions are not exportable
    }
}

pub mod tys {
    pub trait Trait {
        type Type;
    }
    pub struct S;

    impl Trait for S {
        type Type = (u32,);
    }

    #[export_stable]
    pub extern "C" fn foo1(x: <S as Trait>::Type) -> u32 { x.0 }
    //[sdylib]~^ ERROR function with `#[export_stable]` attribute uses type `(u32,)`, which is not exportable

    #[export_stable]
    pub type Type = [i32; 4];

    #[export_stable]
    pub extern "C" fn foo2(_x: Type) {}
    //[sdylib]~^ ERROR function with `#[export_stable]` attribute uses type `[i32; 4]`, which is not exportable

    impl S {
        #[export_stable]
        pub type Type = extern "C" fn();
    }

    #[export_stable]
    pub extern "C" fn foo3(_x: S::Type) {}
    //[sdylib]~^ ERROR function with `#[export_stable]` attribute uses type `extern "C" fn()`, which is not exportable

    #[export_stable]
    pub extern "C" fn foo4() -> impl Copy {
    //[sdylib]~^ ERROR function with `#[export_stable]` attribute uses type `impl Copy`, which is not exportable
        0
    }
}

pub mod privacy {
    #[export_stable]
    #[repr(C)]
    pub struct S1 {
        pub x: i32
    }

    #[export_stable]
    #[repr(C)]
    pub struct S2 {
    //[sdylib]~^ ERROR ADT types with private fields are not exportable
        x: i32
    }

    #[export_stable]
    #[repr(i32)]
    enum E {
    //[sdylib]~^ ERROR private items are not exportable
        Variant1 { x: i32 }
    }
}

pub mod use_site {
    #[export_stable]
    //~^ ERROR cannot be used on
    pub trait Trait {}

    #[export_stable]
    //~^ ERROR cannot be used on
    pub const C: i32 = 0;
}

fn main() {}
