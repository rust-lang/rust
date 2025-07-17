//@ compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0

#![crate_type = "lib"]
// FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
#![feature(rustc_attrs)]
#![feature(fn_align)]

// CHECK: align 16
#[no_mangle]
#[rustc_align(16)]
pub fn fn_align() {}

pub struct A;

impl A {
    // CHECK: align 16
    #[no_mangle]
    #[rustc_align(16)]
    pub fn method_align(self) {}

    // CHECK: align 16
    #[no_mangle]
    #[rustc_align(16)]
    pub fn associated_fn() {}
}

trait T: Sized {
    fn trait_fn() {}

    // CHECK: align 32
    #[rustc_align(32)]
    fn trait_method(self) {}
}

impl T for A {
    // CHECK: align 16
    #[no_mangle]
    #[rustc_align(16)]
    fn trait_fn() {}

    // CHECK: align 16
    #[no_mangle]
    #[rustc_align(16)]
    fn trait_method(self) {}
}

impl T for () {}

pub fn foo() {
    ().trait_method();
}

// CHECK-LABEL: align_specified_twice_1
// CHECK-SAME: align 64
#[no_mangle]
#[rustc_align(32)]
#[rustc_align(64)]
pub fn align_specified_twice_1() {}

// CHECK-LABEL: align_specified_twice_2
// CHECK-SAME: align 128
#[no_mangle]
#[rustc_align(128)]
#[rustc_align(32)]
pub fn align_specified_twice_2() {}

// CHECK-LABEL: align_specified_twice_3
// CHECK-SAME: align 256
#[no_mangle]
#[rustc_align(32)]
#[rustc_align(256)]
pub fn align_specified_twice_3() {}
