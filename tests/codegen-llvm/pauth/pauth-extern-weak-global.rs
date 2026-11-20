// ignore-tidy-file-linelength
//@ add-minicore
//@ revisions: DISC NO_DISC NO_PAUTH

//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination
//@ [NO_PAUTH] needs-llvm-components: aarch64
//@ [NO_PAUTH] compile-flags: --target=aarch64-unknown-linux-gnu --crate-type=lib

// An `extern_weak` static of type `Option<extern "C" fn>` is lowered to an internal global
// initialised with the address of the weak function, or null if the function is not defined at
// link time. Calls through the static authenticate with the discriminator of its function pointer
// type, so the address is signed with that discriminator; without type discrimination, with 0.
//
// Discriminators: `extern "C" fn(i32)` = 2712, `extern "C" fn() -> i64` = 42271.

#![feature(no_core, lang_items, linkage)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

extern "C" {
    #[linkage = "extern_weak"]
    #[link_name = "weak_cb"]
    static WEAK_CB: Option<unsafe extern "C" fn(i32)>;

    #[linkage = "extern_weak"]
    #[link_name = "weak_getter"]
    static WEAK_GETTER: Option<unsafe extern "C" fn() -> i64>;
}

// DISC-DAG: @_rust_extern_with_linkage_{{[0-9a-f]+}}_weak_cb = internal {{.*}}global ptr ptrauth (ptr @weak_cb, i32 0, i64 2712)
// DISC-DAG: @_rust_extern_with_linkage_{{[0-9a-f]+}}_weak_getter = internal {{.*}}global ptr ptrauth (ptr @weak_getter, i32 0, i64 42271)
// NO_DISC-DAG: @_rust_extern_with_linkage_{{[0-9a-f]+}}_weak_cb = internal {{.*}}global ptr ptrauth (ptr @weak_cb, i32 0)
// NO_DISC-DAG: @_rust_extern_with_linkage_{{[0-9a-f]+}}_weak_getter = internal {{.*}}global ptr ptrauth (ptr @weak_getter, i32 0)
// NO_PAUTH-DAG: @_rust_extern_with_linkage_{{[0-9a-f]+}}_weak_cb = internal {{.*}}global ptr @weak_cb
// NO_PAUTH-DAG: @_rust_extern_with_linkage_{{[0-9a-f]+}}_weak_getter = internal {{.*}}global ptr @weak_getter

// CHECK-DAG: declare extern_weak {{.*}}@weak_cb(
// CHECK-DAG: declare extern_weak {{.*}}@weak_getter(

#[no_mangle]
#[allow(unused_unsafe)]
pub fn weak_cb_addr() -> *const Option<unsafe extern "C" fn(i32)> {
    unsafe { &raw const WEAK_CB }
}

#[no_mangle]
#[allow(unused_unsafe)]
pub fn weak_getter_addr() -> *const Option<unsafe extern "C" fn() -> i64> {
    unsafe { &raw const WEAK_GETTER }
}
