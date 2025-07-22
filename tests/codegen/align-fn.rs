//@ compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0 -Clink-dead-code
//@ edition: 2024
//@ ignore-wasm32 aligning functions is not currently supported on wasm (#143368)

#![crate_type = "lib"]
// FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
#![feature(rustc_attrs)]
#![feature(fn_align)]

// CHECK: align 16
#[unsafe(no_mangle)]
#[rustc_align(16)]
pub fn fn_align() {}

pub struct A;

impl A {
    // CHECK: align 16
    #[unsafe(no_mangle)]
    #[rustc_align(16)]
    pub fn method_align(self) {}

    // CHECK: align 16
    #[unsafe(no_mangle)]
    #[rustc_align(16)]
    pub fn associated_fn() {}
}

trait T: Sized {
    fn trait_fn() {}

    fn trait_method(self) {}

    #[rustc_align(8)]
    fn trait_method_inherit_low(self);

    #[rustc_align(32)]
    fn trait_method_inherit_high(self);

    #[rustc_align(32)]
    fn trait_method_inherit_default(self) {}

    #[rustc_align(4)]
    #[rustc_align(128)]
    #[rustc_align(8)]
    fn inherit_highest(self) {}
}

impl T for A {
    // CHECK-LABEL: trait_fn
    // CHECK-SAME: align 16
    #[unsafe(no_mangle)]
    #[rustc_align(16)]
    fn trait_fn() {}

    // CHECK-LABEL: trait_method
    // CHECK-SAME: align 16
    #[unsafe(no_mangle)]
    #[rustc_align(16)]
    fn trait_method(self) {}

    // The prototype's align is ignored because the align here is higher.
    // CHECK-LABEL: trait_method_inherit_low
    // CHECK-SAME: align 16
    #[unsafe(no_mangle)]
    #[rustc_align(16)]
    fn trait_method_inherit_low(self) {}

    // The prototype's align is used because it is higher.
    // CHECK-LABEL: trait_method_inherit_high
    // CHECK-SAME: align 32
    #[unsafe(no_mangle)]
    #[rustc_align(16)]
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
    #[rustc_align(32)]
    #[rustc_align(64)]
    fn inherit_highest(self) {}
}

trait HasDefaultImpl: Sized {
    // CHECK-LABEL: inherit_from_default_method
    // CHECK-LABEL: inherit_from_default_method
    // CHECK-SAME: align 32
    #[rustc_align(32)]
    fn inherit_from_default_method(self) {}
}

pub struct InstantiateDefaultMethods;

impl HasDefaultImpl for InstantiateDefaultMethods {}

// CHECK-LABEL: align_specified_twice_1
// CHECK-SAME: align 64
#[unsafe(no_mangle)]
#[rustc_align(32)]
#[rustc_align(64)]
pub fn align_specified_twice_1() {}

// CHECK-LABEL: align_specified_twice_2
// CHECK-SAME: align 128
#[unsafe(no_mangle)]
#[rustc_align(128)]
#[rustc_align(32)]
pub fn align_specified_twice_2() {}

// CHECK-LABEL: align_specified_twice_3
// CHECK-SAME: align 256
#[unsafe(no_mangle)]
#[rustc_align(32)]
#[rustc_align(256)]
pub fn align_specified_twice_3() {}

const _: () = {
    // CHECK-LABEL: align_unmangled
    // CHECK-SAME: align 256
    #[unsafe(no_mangle)]
    #[rustc_align(32)]
    #[rustc_align(256)]
    extern "C" fn align_unmangled() {}
};

unsafe extern "C" {
    #[rustc_align(256)]
    fn align_unmangled();
}

// FIXME also check `gen` et al
// CHECK-LABEL: async_align
// CHECK-SAME: align 64
#[unsafe(no_mangle)]
#[rustc_align(64)]
pub async fn async_align() {}
