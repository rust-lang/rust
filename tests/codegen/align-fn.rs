//@ compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0 -Clink-dead-code

#![crate_type = "lib"]
#![feature(fn_align)]

// CHECK: align 16
#[unsafe(no_mangle)]
#[align(16)]
pub fn fn_align() {}

pub struct A;

impl A {
    // CHECK: align 16
    #[unsafe(no_mangle)]
    #[align(16)]
    pub fn method_align(self) {}

    // CHECK: align 16
    #[unsafe(no_mangle)]
    #[align(16)]
    pub fn associated_fn() {}
}

trait T: Sized {
    fn trait_fn() {}

    fn trait_method(self) {}

    #[align(8)]
    fn trait_method_inherit_low(self);

    #[align(32)]
    fn trait_method_inherit_high(self);

    #[align(32)]
    fn trait_method_inherit_default(self) {}

    #[align(4)]
    #[align(128)]
    #[align(8)]
    fn inherit_highest(self) {}
}

impl T for A {
    // CHECK-LABEL: trait_fn
    // CHECK-SAME: align 16
    #[unsafe(no_mangle)]
    #[align(16)]
    fn trait_fn() {}

    // CHECK-LABEL: trait_method
    // CHECK-SAME: align 16
    #[unsafe(no_mangle)]
    #[align(16)]
    fn trait_method(self) {}

    // The prototype's align is ignored because the align here is higher.
    // CHECK-LABEL: trait_method_inherit_low
    // CHECK-SAME: align 16
    #[unsafe(no_mangle)]
    #[align(16)]
    fn trait_method_inherit_low(self) {}

    // The prototype's align is used because it is higher.
    // CHECK-LABEL: trait_method_inherit_high
    // CHECK-SAME: align 32
    #[unsafe(no_mangle)]
    #[align(16)]
    fn trait_method_inherit_high(self) {}

    // The prototype's align inherited.
    // CHECK-LABEL: trait_method_inherit_default
    // CHECK-SAME: align 32
    #[unsafe(no_mangle)]
    fn trait_method_inherit_default(self) {}

    // The prototype's highest align inherited.
    // CHECK-LABEL: inherit_highest
    // CHECK-SAME: align 128
    #[unsafe(no_mangle)]
    #[align(32)]
    #[align(64)]
    fn inherit_highest(self) {}
}

trait HasDefaultImpl: Sized {
    // CHECK-LABEL: inherit_from_default_method
    // CHECK-LABEL: inherit_from_default_method
    // CHECK-SAME: align 32
    #[align(32)]
    fn inherit_from_default_method(self) {}
}

pub struct InstantiateDefaultMethods;

impl HasDefaultImpl for InstantiateDefaultMethods {}

// CHECK-LABEL: align_specified_twice_1
// CHECK-SAME: align 64
#[unsafe(no_mangle)]
#[align(32)]
#[align(64)]
pub fn align_specified_twice_1() {}

// CHECK-LABEL: align_specified_twice_2
// CHECK-SAME: align 128
#[unsafe(no_mangle)]
#[align(128)]
#[align(32)]
pub fn align_specified_twice_2() {}

// CHECK-LABEL: align_specified_twice_3
// CHECK-SAME: align 256
#[unsafe(no_mangle)]
#[align(32)]
#[align(256)]
pub fn align_specified_twice_3() {}
