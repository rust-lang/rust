// Check that an implicit default `-Ttarget-cpu` and an explicit default
// `-Ttarget-cpu` compare equal.
//
// NVPTX requires consistent `-Ttarget-cpu` values across crates, but it does
// not require the CPU to be specified explicitly. Therefore, compiling one crate
// without `-Ttarget-cpu` and another crate with the target's default CPU
// explicitly specified must be accepted.
//
// The mismatch revisions additionally check that an implicit or explicit default
// CPU must not match a different explicitly specified CPU.

//@ aux-build:target_cpu_default_implicit.rs
//@ aux-build:target_cpu_default_explicit.rs
//@ aux-build:target_cpu_non_default.rs
//@ compile-flags: --target nvptx64-nvidia-cuda
//@ needs-llvm-components: nvptx
//@ ignore-backends: gcc

//@ revisions: implicit_default explicit_default implicit_mismatch explicit_mismatch
//@[implicit_default] check-pass
//@[explicit_default] compile-flags: -Ttarget-cpu=sm_70
//@[explicit_default] check-pass
//@[explicit_mismatch] compile-flags: -Ttarget-cpu=sm_70

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

#[cfg(any(implicit_default, explicit_default))]
extern crate target_cpu_default_implicit;

#[cfg(any(implicit_default, explicit_default))]
extern crate target_cpu_default_explicit;

#[cfg(any(implicit_mismatch, explicit_mismatch))]
extern crate target_cpu_non_default;

//[implicit_mismatch]~? ERROR mixing `-Ttarget-cpu` will cause an ABI mismatch
//[explicit_mismatch]~? ERROR mixing `-Ttarget-cpu` will cause an ABI mismatch
