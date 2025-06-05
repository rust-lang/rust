//@ run-pass
//@ compile-flags: -C codegen-units=3
//@ aux-build:sepcomp-extern-lib.rs

// Test accessing external items from multiple compilation units.

extern crate sepcomp_extern_lib;

extern "C" {
    fn foo() -> usize;
}

fn call1() -> usize {
    unsafe { foo() }
}

mod a {
    pub fn call2() -> usize {
        unsafe { crate::foo() }
    }
}

mod b {
    pub fn call3() -> usize {
        unsafe { crate::foo() }
    }
}

fn main() {
    assert_eq!(call1(), 1234);
    assert_eq!(a::call2(), 1234);
    assert_eq!(b::call3(), 1234);
}
