//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: --target=nvptx64-nvidia-cuda --crate-type cdylib
//@ needs-llvm-components: nvptx

#![feature(abi_ptx, no_core, intrinsics)]
#![no_core]

extern crate minicore;
use minicore::*;

#[rustc_intrinsic]
pub const fn wrapping_mul<T: Copy>(a: T, b: T) -> T;

// We want to make sure neither we nor LLVM introduce characters invalid for
// NVPTX.
// CHECK: .func (.param .b32 func_retval0) [[IMPL_FN:[a-zA-Z0-9$_]+square]]

// CHECK-LABEL: .visible .entry top_kernel(
#[no_mangle]
pub unsafe extern "ptx-kernel" fn top_kernel(a: *const u32, b: *mut u32) {
    // CHECK:      call.uni (retval0),
    // CHECK-SAME: [[IMPL_FN]]
    *b = deep::private::MyStruct::new(*a).square();
}

pub mod deep {
    pub mod private {
        use crate::wrapping_mul;
        pub struct MyStruct<T>(T);

        impl MyStruct<u32> {
            pub fn new(a: u32) -> Self {
                MyStruct(a)
            }

            #[inline(never)]
            pub fn square(&self) -> u32 {
                wrapping_mul(self.0, self.0)
            }
        }
    }
}
