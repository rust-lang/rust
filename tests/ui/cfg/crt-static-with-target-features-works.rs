// Test to ensure that specifying a value for crt-static in target features
// does not result in skipping the features following it.
// This is a regression test for #144143

//@ add-core-stubs
//@ needs-llvm-components: x86
//@ compile-flags: --target=x86_64-unknown-linux-gnu
//@ compile-flags: -Ctarget-feature=+crt-static,+avx2

#![crate_type = "rlib"]
#![feature(no_core, rustc_attrs, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[rustc_builtin_macro]
macro_rules! compile_error {
    () => {};
}

#[cfg(target_feature = "avx2")]
compile_error!("+avx2");
//~^ ERROR: +avx2
