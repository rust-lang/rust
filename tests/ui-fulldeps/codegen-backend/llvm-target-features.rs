//! When using llvm-target-features, we cannot mix codegen backends.

//@ ignore-stage1 (requires matching sysroot built with in-tree compiler)
//@ ignore-backends: gcc

//@ aux-codegen-backend: the_backend.rs
//@ aux-build: with_llvm_target_feature.rs

// Pick a target that requires no target features in the ABI check, so that no warning is shown
// about missing target features.
//@ compile-flags: --target arm-unknown-linux-gnueabi --crate-type=lib
//@ needs-llvm-components: arm
//@ compile-flags: -Zllvm-target-feature=+thumb2

#![feature(no_core)]
#![no_core]

extern crate with_llvm_target_feature;

//~? ERROR: mixing `-Zcodegen-backend` will cause an ABI mismatch
