// Checks that the gpu-kernel calling convention correctly translates to LLVM calling conventions.

//@ revisions: nvptx
//@ [nvptx] compile-flags: --crate-type=rlib --target=nvptx64-nvidia-cuda
//@ [nvptx] needs-llvm-components: nvptx
#![feature(no_core, lang_items, abi_gpu_kernel)]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "freeze"]
trait Freeze {}
#[lang = "copy"]
trait Copy {}

// nvptx: define ptx_kernel void @fun(i32
#[no_mangle]
pub extern "gpu-kernel" fn fun(_: i32) {}
