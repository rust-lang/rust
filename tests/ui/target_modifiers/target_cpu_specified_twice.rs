//@ aux-build:target_cpu_is_target_modifier.rs
//@ compile-flags: --target nvptx64-nvidia-cuda
//@ needs-llvm-components: nvptx

//@ revisions:last_correct last_incorrect_allow error_generated
//@[last_correct] compile-flags: -Ctarget-cpu=sm_70 -Ctarget-cpu=sm_60
//@[last_incorrect_allow] compile-flags: -Ctarget-cpu=sm_60 -Ctarget-cpu=sm_70
//@[last_incorrect_allow] compile-flags: -Cunsafe-allow-abi-mismatch=target-cpu
//@[error_generated] compile-flags: -Ctarget-cpu=sm_60 -Ctarget-cpu=sm_70
//@[last_correct] check-pass
//@[last_incorrect_allow] check-pass
//@ ignore-backends: gcc

#![feature(no_core)]
//[error_generated]~^ ERROR mixing `-Ctarget-cpu` will cause an ABI mismatch in crate
// `target_cpu_specified_twice`
#![crate_type = "rlib"]
#![no_core]

extern crate target_cpu_is_target_modifier;
