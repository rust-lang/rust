// This test checks that we don't lint values defined by a custom target (target json)
//
//@ add-minicore
//@ check-pass
//@ no-auto-check-cfg
//@ needs-llvm-components: x86
//@ compile-flags: --crate-type=lib --check-cfg=cfg() --target={{src-base}}/check-cfg/my-awesome-platform.json
//@ ignore-backends: gcc

#![feature(lang_items, no_core, auto_traits, rustc_attrs)]
#![no_core]

extern crate minicore;
use minicore::*;

#[rustc_builtin_macro]
macro_rules! compile_error {
    () => {};
}

#[cfg(not(target_os = "ericos"))]
compile_error!("target_os from target JSON not wired through");

#[cfg(not(target_arch = "tamirdarch"))]
compile_error!("target_arch from target JSON not wired through");
