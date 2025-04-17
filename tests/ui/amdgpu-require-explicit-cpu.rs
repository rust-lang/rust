//@ revisions: nocpu cpu
//@ no-prefer-dynamic
//@ compile-flags: --crate-type=cdylib --target=amdgcn-amd-amdhsa
//@ needs-llvm-components: amdgpu
//@ needs-rust-lld
//@[nocpu] build-fail
//@[cpu] compile-flags: -Ctarget-cpu=gfx900
//@[cpu] build-pass

#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

pub fn foo() {}

//[nocpu]~? ERROR target requires explicitly specifying a cpu with `-C target-cpu`
