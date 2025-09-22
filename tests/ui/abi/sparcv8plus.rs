//@ add-core-stubs
//@ revisions: sparc sparcv8plus sparc_cpu_v9 sparc_feature_v8plus sparc_cpu_v9_feature_v8plus
//@[sparc] compile-flags: --target sparc-unknown-none-elf
//@[sparc] needs-llvm-components: sparc
//@[sparcv8plus] compile-flags: --target sparc-unknown-linux-gnu
//@[sparcv8plus] needs-llvm-components: sparc
//@[sparc_cpu_v9] compile-flags: --target sparc-unknown-none-elf -C target-cpu=v9
//@[sparc_cpu_v9] needs-llvm-components: sparc
//@[sparc_feature_v8plus] compile-flags: --target sparc-unknown-none-elf -C target-feature=+v8plus
//@[sparc_feature_v8plus] needs-llvm-components: sparc
//@[sparc_cpu_v9_feature_v8plus] compile-flags: --target sparc-unknown-none-elf -C target-cpu=v9 -C target-feature=+v8plus
//@[sparc_cpu_v9_feature_v8plus] needs-llvm-components: sparc

#![crate_type = "rlib"]
#![feature(no_core, rustc_attrs, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[rustc_builtin_macro]
macro_rules! compile_error {
    () => {};
}

#[cfg(all(not(target_feature = "v8plus"), not(target_feature = "v9")))]
compile_error!("-v8plus,-v9");
//[sparc]~^ ERROR -v8plus,-v9

#[cfg(all(target_feature = "v8plus", target_feature = "v9"))]
compile_error!("+v8plus,+v9");
//[sparcv8plus,sparc_cpu_v9_feature_v8plus]~^ ERROR +v8plus,+v9

// FIXME: should be rejected
#[cfg(all(target_feature = "v8plus", not(target_feature = "v9")))]
compile_error!("+v8plus,-v9 (FIXME)");
//[sparc_feature_v8plus]~^ ERROR +v8plus,-v9 (FIXME)

#[cfg(all(not(target_feature = "v8plus"), target_feature = "v9"))]
compile_error!("-v8plus,+v9");
//[sparc_cpu_v9]~^ ERROR -v8plus,+v9
