//! Check that certain target *respect* the unsupported-cpus in `-C target-cpu`.

//@ revisions: nvptx-sm60

//@[nvptx-sm60] compile-flags: --target=nvptx64-nvidia-cuda --crate-type=rlib -Ctarget-cpu=sm_60
//@[nvptx-sm60] needs-llvm-components: nvptx
//@[nvptx-sm60] build-fail
//@ ignore-backends: gcc

#![feature(no_core)]
#![no_core]
#![crate_type = "rlib"]

//[nvptx-sm60]~? ERROR target cpu `sm_60` is known but unsupported
