//@ revisions: HOST AMDGPU NVPTX
//@ add-core-stubs
//@ compile-flags: --crate-type=rlib
//@[AMDGPU] compile-flags: --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx1100
//@[AMDGPU] needs-llvm-components: amdgpu
//@[NVPTX]  compile-flags: --target nvptx64-nvidia-cuda
//@[NVPTX] needs-llvm-components: nvptx

#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// Functions
extern "gpu-kernel" fn f1(_: ()) {} //~ ERROR "gpu-kernel" ABI is experimental and subject to change
//[HOST]~^ ERROR is not a supported ABI

// Methods in trait definition
trait Tr {
    extern "gpu-kernel" fn m1(_: ()); //~ ERROR "gpu-kernel" ABI is experimental and subject to change

    extern "gpu-kernel" fn dm1(_: ()) {} //~ ERROR "gpu-kernel" ABI is experimental and subject to change
    //[HOST]~^ ERROR is not a supported ABI
}

struct S;

// Methods in trait impl
impl Tr for S {
    extern "gpu-kernel" fn m1(_: ()) {} //~ ERROR "gpu-kernel" ABI is experimental and subject to change
    //[HOST]~^ ERROR is not a supported ABI
}

// Methods in inherent impl
impl S {
    extern "gpu-kernel" fn im1(_: ()) {} //~ ERROR "gpu-kernel" ABI is experimental and subject to change
    //[HOST]~^ ERROR is not a supported ABI
}

// Function pointer types
type A1 = extern "gpu-kernel" fn(_: ()); //~ ERROR "gpu-kernel" ABI is experimental and subject to change
//[HOST]~^ WARNING the calling convention "gpu-kernel" is not supported on this target [unsupported_fn_ptr_calling_conventions]
//[HOST]~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

// Foreign modules
extern "gpu-kernel" {} //~ ERROR "gpu-kernel" ABI is experimental and subject to change
//[HOST]~^ ERROR is not a supported ABI
