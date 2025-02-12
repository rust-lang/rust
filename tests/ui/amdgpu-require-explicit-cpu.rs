//@ revisions: nocpu cpu
//@ no-prefer-dynamic
//@ compile-flags: --crate-type=cdylib --target=amdgcn-amd-amdhsa
//@ needs-llvm-components: amdgpu
//@ needs-rust-lld
//@[nocpu] error-pattern: target requires explicitly specifying a cpu
//@[nocpu] build-fail
//@[cpu] compile-flags: -Ctarget-cpu=gfx900
//@[cpu] build-pass

#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized {}

pub fn foo() {}
