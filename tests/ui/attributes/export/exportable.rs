//@ compile-flags: -Zunstable-options -Csymbol-mangling-version=v0

#![crate_type = "dylib"]
#![allow(incomplete_features)]
#![feature(export)]
#![feature(inherent_associated_types)]

mod m {
    #[export]
    pub struct S;
    //~^ ERROR private items are not exportable

    pub fn foo() {}
    //~^ ERROR Only `extern "C"` functions are exportable
}

#[export]
pub use m::foo;

#[export]
pub mod m1 {
    #[repr(C)]
    pub struct S1; // OK, public type with stable repr

    struct S2;

    pub struct S3;
    //~^ ERROR types with unstable layout are not exportable
}

pub mod fn_sig {
    #[export]
    pub fn foo1() {}
    //~^ ERROR Only `extern "C"` functions are exportable

    #[export]
    #[repr(C)]
    pub struct S;

    #[export]
    pub extern "C" fn foo2(x: S) {}

    #[export]
    pub extern "C" fn foo3(x: Box<S>) {}
    //~^ ERROR function with `#[export]` attribute uses type `Box<fn_sig::S>`, which is not exportable
}

pub mod impl_item {
    pub struct S;

    impl S {
        #[export]
        pub extern "C" fn foo1(&self) {}
        //~^ ERROR method with `#[export]` attribute uses type `&impl_item::S`, which is not exportable

        #[export]
        pub extern "C" fn foo2(self) {}
        //~^ ERROR method with `#[export]` attribute uses type `impl_item::S`, which is not exportable
    }

    pub struct S2<T>(T);

    impl<T> S2<T> {
        #[export]
        pub extern "C" fn foo1(&self) {}
        //~^ ERROR generic method's are not exportable
    }
}

pub mod tys {
    trait Trait {
        type Type;
    }
    pub struct S;

    impl Trait for S {
        type Type = (u32,);
    }

    #[export]
    pub extern "C" fn foo1(x: <S as Trait>::Type) {}
    //~^ ERROR function with `#[export]` attribute uses type `(u32,)`, which is not exportable

    #[export]
    pub type Type = [i32; 4];
    //~^ ERROR type alias with `#[export]` attribute uses type `[i32; 4]`, which is not exportable

    impl S {
        #[export]
        pub type Type = extern "C" fn();
        //~^ ERROR associated type with `#[export]` attribute uses type `extern "C" fn()`, which is not exportable
    }

    struct S1;

    #[export]
    #[repr(C)]
    pub struct S2 {
    //~^ ERROR struct with `#[export]` attribute uses type `tys::S1`, which is not exportable
        pub x: S1,
    }
}

pub mod privacy {
    #[export]
    #[repr(C)]
    pub struct S1 {
        pub x: i32
    }

    #[export]
    #[repr(C)]
    pub struct S2 {
    //~^ ERROR ADT types with private fields are not exportable
        x: i32
    }

    #[export]
    #[repr(i32)]
    enum E {
    //~^ ERROR private items are not exportable
        Variant1 { x: i32 }
    }
}

pub mod use_site {
    #[export]
    pub trait Trait {}
    //~^ ERROR trait's are not exportable

    #[export]
    pub const C: i32 = 0;
    //~^ ERROR constant's are not exportable
}

fn main() {}
