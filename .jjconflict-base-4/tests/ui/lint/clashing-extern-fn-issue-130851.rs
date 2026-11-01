//@ build-pass
#![warn(clashing_extern_declarations)]

#[repr(C)]
pub struct A {
    a: [u16; 4],
}
#[repr(C)]
pub struct B {
    b: [u32; 4],
}

pub mod a {
    extern "C" {
        pub fn foo(_: super::A);
    }
}
pub mod b {
    extern "C" {
        pub fn foo(_: super::B);
        //~^ WARN `foo` redeclared with a different signature
    }
}

#[repr(C)]
pub struct G<T> {
    g: [T; 4],
}

pub mod x {
    extern "C" {
        pub fn bar(_: super::G<u16>);
    }
}
pub mod y {
    extern "C" {
        pub fn bar(_: super::G<u32>);
        //~^ WARN `bar` redeclared with a different signature
    }
}

fn main() {}
