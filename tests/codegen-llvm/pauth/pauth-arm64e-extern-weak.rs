//@ add-minicore
//@ needs-llvm-components: aarch64
//@ compile-flags: --target=arm64e-apple-ios -Copt-level=0 -Cpanic=abort

#![feature(no_core, lang_items, linkage)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[lang = "Option"]
pub enum WeakOption<T> {
    None,
    Some(T),
}

extern "Rust" {
    #[linkage = "extern_weak"]
    static weak_rust: WeakOption<unsafe fn()>;
}

// CHECK: @_rust_extern_with_linkage_{{[0-9a-f]+}}_weak_rust = internal {{.*}}global ptr ptrauth (ptr @weak_rust, i32 0)
// CHECK-LABEL: define {{.*}} @weak_address(
// CHECK: ret ptr @_rust_extern_with_linkage_{{[0-9a-f]+}}_weak_rust
#[no_mangle]
pub unsafe fn weak_address() -> *const WeakOption<unsafe fn()> {
    &raw const weak_rust
}
