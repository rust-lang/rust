// This test checks that we don't lint values defined by a custom target (target json)
//
//@ add-core-stubs
//@ check-pass
//@ no-auto-check-cfg
//@ needs-llvm-components: x86
//@ compile-flags: --crate-type=lib --check-cfg=cfg() --target={{src-base}}/check-cfg/my-awesome-platform.json

#![feature(lang_items, no_core, auto_traits)]
#![no_core]

extern crate minicore;
use minicore::*;

#[cfg(target_os = "linux")]
fn target_os_linux() {}

#[cfg(target_os = "ericos")]
fn target_os_ericos() {}
