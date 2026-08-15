// Check that GPU targets do not run jump-threading

//@ ignore-backends: gcc
//@ add-minicore
//@ revisions: cpu gpu
//@ compile-flags: -Z mir-opt-level=4
//@[cpu] compile-flags: --target x86_64-unknown-linux-gnu
//@[cpu] needs-llvm-components: x86
//@[gpu] compile-flags: --target nvptx64-nvidia-cuda
//@[gpu] needs-llvm-components: nvptx

#![crate_type = "lib"]
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

#[inline(never)]
fn opaque() {}

#[inline(never)]
fn opaque2() {}

#[inline(never)]
fn syncthreads() {}

pub fn function(cond: bool) {
    // CHECK-LABEL: fn function
    // Jump-threading duplicates syncthreads
    // cpu: syncthreads()
    // cpu: syncthreads()

    // Must not duplicate syncthreads
    // gpu: syncthreads()
    // gpu-NOT: syncthreads()

    if cond {
        opaque();
    } else {
        opaque2();
    }
    syncthreads();
    if cond {
        opaque();
    } else {
        opaque2();
    }
}
