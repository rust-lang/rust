//@ compile-flags: --target=s390x-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: systemz
//@ ignore-backends: gcc
//@ add-minicore
#![feature(no_core)]
#![no_core]
extern crate minicore;
use minicore::*;

// `sse2` only exists on x86 — single-arch cross-arch note
#[target_feature(enable = "sse2")]
//~^ ERROR not valid for this target
//~| NOTE `sse2` is not valid for this target
//~| HELP valid names are:
//~| NOTE `sse2` is present on the `x86` target architecture. Did you mean to compile for that target, or use conditional compilation?
unsafe fn x86_feature_on_s390x() {}

// `aes` exists on arm, aarch64, and x86 — multi-arch cross-arch note
#[target_feature(enable = "aes")]
//~^ ERROR not valid for this target
//~| NOTE `aes` is not valid for this target
//~| HELP valid names are:
//~| NOTE `aes` is present on the `aarch64`, `arm`, and `x86` target architectures. Did you mean to compile for one of those targets, or use conditional compilation?
unsafe fn multi_arch_feature_on_s390x() {}

// Bogus feature doesn't exist on any arch — no cross-arch note
#[target_feature(enable = "nonesuch")]
//~^ ERROR not valid for this target
//~| NOTE `nonesuch` is not valid for this target
//~| HELP valid names are:
unsafe fn bogus_feature_on_s390x() {}
