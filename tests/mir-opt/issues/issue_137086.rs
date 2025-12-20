// Check that GPU targets do not run jump-threading

//@ revisions: cpu gpu
//@ compile-flags: -Z mir-opt-level=4
//@[cpu] compile-flags: --target x86_64-unknown-linux-gnu
//@[cpu] needs-llvm-components: x86
//@[gpu] compile-flags: --target nvptx64-nvidia-cuda
//@[gpu] needs-llvm-components: nvptx

#![crate_type = "lib"]
#![feature(no_core, lang_items, rustc_attrs)]
#![no_std]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}
#[lang = "copy"]
trait Copy {}

impl Copy for bool {}

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
    // gpu-not: syncthreads()

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
