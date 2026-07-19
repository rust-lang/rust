//@ aux-build:wrong_llvm_target_feature.rs
//@ compile-flags: --target x86_64-unknown-linux-gnu -Zllvm-target-feature=+another-feature
//@ needs-llvm-components: x86

//@ revisions:allow_llvm_target_feature_mismatch allow_no_value error_generated
//@[allow_llvm_target_feature_mismatch] compile-flags: -Cunsafe-allow-abi-mismatch=llvm-target-feature
//@[allow_no_value] compile-flags: -Cunsafe-allow-abi-mismatch
// [error_generated] no extra compile-flags
//@[allow_llvm_target_feature_mismatch] check-pass
//@ ignore-backends: gcc

#![feature(no_core)]
//[error_generated]~^ ERROR mixing `-Zllvm-target-feature` will cause an ABI mismatch in crate `incompatible_llvm_target_feature`
#![crate_type = "rlib"]
#![no_core]

extern crate wrong_llvm_target_feature;

//[allow_no_value]~? ERROR codegen option `unsafe-allow-abi-mismatch` requires a comma-separated list of strings
