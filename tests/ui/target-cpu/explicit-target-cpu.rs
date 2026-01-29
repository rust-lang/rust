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

//@ revisions: nvptx_nocpu nvptx_cpu

//@[nvptx_nocpu] compile-flags: --target=nvptx64-nvidia-cuda
//@[nvptx_nocpu] needs-llvm-components: nvptx
//@[nvptx_nocpu] build-fail

//@[nvptx_cpu] compile-flags: --target=nvptx64-nvidia-cuda
//@[nvptx_cpu] needs-llvm-components: nvptx
//@[nvptx_cpu] compile-flags: -Ctarget-cpu=sm_30
//@[nvptx_cpu] build-pass

//@ ignore-backends: gcc

#![crate_type = "rlib"]
// We don't want to link in any other crate as this would make it necessary to specify
// a `-Ctarget-cpu` for them resulting in a *target-modifier* disagreement error instead of the
// error mentioned below.
#![feature(no_core)]
#![no_core]

//[amdgcn_nocpu,avr_nocpu,nvptx_nocpu]~? ERROR target requires explicitly specifying a cpu with `-C target-cpu`
