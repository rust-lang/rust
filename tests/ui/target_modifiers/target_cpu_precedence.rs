// Check that when `-Ctarget-cpu` is specified multiple times, its last occurence
// takes precedence to populate the crate metadata.
// Therefore, the target modifier comparison also uses this value.

// The auxiliary crate is compiled with `-Ctarget-cpu=sm_80`. Revisions where
// the last occurence of `-Ctarget-cpu` is `sm_80` must succeed. `mismatch_allowed`
// must also succeed because `-Cunsafe-allow-abi-mismatch=target-cpu`
// is set. `mismatch_error` must fail.
//
//@ aux-build:target_cpu_non_default.rs
//@ compile-flags: --target nvptx64-nvidia-cuda
//@ needs-llvm-components: nvptx

//@ revisions: last_matches mismatch_allowed mismatch_error
//@[last_matches] compile-flags: -Ctarget-cpu=sm_90 -Ctarget-cpu=sm_80
//@[mismatch_allowed] compile-flags: -Ctarget-cpu=sm_80 -Ctarget-cpu=sm_90
//@[mismatch_allowed] compile-flags: -Cunsafe-allow-abi-mismatch=target-cpu
//@[mismatch_error] compile-flags: -Ctarget-cpu=sm_80 -Ctarget-cpu=sm_90
//@[last_matches] check-pass
//@[mismatch_allowed] check-pass
//@ ignore-backends: gcc

#![feature(no_core)]
//[mismatch_error]~^ ERROR mixing `-Ctarget-cpu` will cause an ABI mismatch in crate
// `target_cpu_precedence`
#![crate_type = "rlib"]
#![no_core]

extern crate target_cpu_non_default;
