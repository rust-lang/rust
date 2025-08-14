//! Check that certain target *requires* the user to specify a target CPU via `-C target-cpu`.

//@ revisions: amdgcn_nocpu amdgcn_cpu

//@[amdgcn_nocpu] compile-flags: --target=amdgcn-amd-amdhsa
//@[amdgcn_nocpu] needs-llvm-components: amdgpu
//@[amdgcn_nocpu] build-fail

//@[amdgcn_cpu] compile-flags: --target=amdgcn-amd-amdhsa
//@[amdgcn_cpu] needs-llvm-components: amdgpu
//@[amdgcn_cpu] compile-flags: -Ctarget-cpu=gfx900
//@[amdgcn_cpu] build-pass

//@ revisions: avr_nocpu avr_cpu

//@[avr_nocpu] compile-flags: --target=avr-none
//@[avr_nocpu] needs-llvm-components: avr
//@[avr_nocpu] build-fail

//@[avr_cpu] compile-flags: --target=avr-none
//@[avr_cpu] needs-llvm-components: avr
//@[avr_cpu] compile-flags: -Ctarget-cpu=atmega328p
//@[avr_cpu] build-pass

#![crate_type = "rlib"]

// FIXME(#140038): this can't use `minicore` yet because `minicore` doesn't currently propagate the
// `-C target-cpu` for targets that *require* a `target-cpu` being specified.
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang="sized"]
trait Sized {}

pub fn foo() {}

//[amdgcn_nocpu,avr_nocpu]~? ERROR target requires explicitly specifying a cpu with `-C target-cpu`
