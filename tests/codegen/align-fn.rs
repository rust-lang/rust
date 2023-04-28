// compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0

#![crate_type = "lib"]
#![feature(fn_align)]

// CHECK: align 16
#[no_mangle]
#[repr(align(16))]
pub fn fn_align() {}

pub struct A;

impl A {
    // CHECK: align 16
    #[no_mangle]
    #[repr(align(16))]
    pub fn method_align(self) {}

    // CHECK: align 16
    #[no_mangle]
    #[repr(align(16))]
    pub fn associated_fn() {}
}

trait T: Sized {
    fn trait_fn() {}

    // CHECK: align 32
    #[repr(align(32))]
    fn trait_method(self) {}
}

impl T for A {
    // CHECK: align 16
    #[no_mangle]
    #[repr(align(16))]
    fn trait_fn() {}

    // CHECK: align 16
    #[no_mangle]
    #[repr(align(16))]
    fn trait_method(self) {}
}

impl T for () {}

pub fn foo() {
    ().trait_method();
}
