// Test argument and return type restrictions of the gpu-kernel ABI and
// check for warnings on mangled gpu-kernels.

//@ check-pass
//@ ignore-backends: gcc
//@ revisions: amdgpu nvptx
//@ add-minicore
//@ edition: 2024
//@[amdgpu] compile-flags: --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@[amdgpu] needs-llvm-components: amdgpu
//@[nvptx]  compile-flags: --target nvptx64-nvidia-cuda
//@[nvptx] needs-llvm-components: nvptx

#![feature(no_core, abi_gpu_kernel)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// Return types can be () or !
#[unsafe(no_mangle)]
extern "gpu-kernel" fn ret_empty() {}
#[unsafe(no_mangle)]
extern "gpu-kernel" fn ret_never() -> ! { loop {} }

// Arguments can be scalars or pointers
#[unsafe(no_mangle)]
extern "gpu-kernel" fn arg_i32(_: i32) { }
#[unsafe(no_mangle)]
extern "gpu-kernel" fn arg_ptr(_: *const i32) { }
#[unsafe(no_mangle)]
extern "gpu-kernel" fn arg_ptr_mut(_: *mut i32) { }

#[unsafe(no_mangle)]
extern "gpu-kernel" fn arg_zst(_: ()) { }
//~^ WARN passing type `()` to a function with "gpu-kernel" ABI may have unexpected behavior
//~^^ WARN `extern` fn uses type `()`, which is not FFI-safe

#[unsafe(no_mangle)]
extern "gpu-kernel" fn arg_ref(_: &i32) { }
//~^ WARN passing type `&i32` to a function with "gpu-kernel" ABI may have unexpected behavior
#[unsafe(no_mangle)]
extern "gpu-kernel" fn arg_ref_mut(_: &mut i32) { }
//~^ WARN passing type `&mut i32` to a function with "gpu-kernel" ABI may have unexpected behavior

struct S { a: i32, b: i32 }
#[unsafe(no_mangle)]
extern "gpu-kernel" fn arg_struct(_: S) { }
//~^ WARN passing type `S` to a function with "gpu-kernel" ABI may have unexpected behavior
//~^^ WARN `extern` fn uses type `S`, which is not FFI-safe

#[unsafe(no_mangle)]
extern "gpu-kernel" fn arg_tup(_: (i32, i32)) { }
//~^ WARN passing type `(i32, i32)` to a function with "gpu-kernel" ABI may have unexpected behavior
//~^^ WARN `extern` fn uses type `(i32, i32)`, which is not FFI-safe

#[unsafe(export_name = "kernel")]
pub extern "gpu-kernel" fn allowed_kernel_name() {}

pub extern "gpu-kernel" fn mangled_kernel() { }
//~^ WARN function with the "gpu-kernel" ABI has a mangled name
