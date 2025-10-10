//@ assembly-output: ptx-linker
//@ compile-flags: --crate-type cdylib -Z unstable-options -Clinker-flavor=llbc
//@ only-nvptx64
//@ revisions: LLVM20 LLVM21
//@ [LLVM21] min-llvm-version: 21
//@ [LLVM20] max-llvm-major-version: 20

#![feature(abi_ptx)]
#![no_std]

//@ aux-build: breakpoint-panic-handler.rs
extern crate breakpoint_panic_handler;

// Verify function name doesn't contain unacceaptable characters.
// CHECK: .func (.param .b32 func_retval0) [[IMPL_FN:[a-zA-Z0-9$_]+square[a-zA-Z0-9$_]+]]

// CHECK-LABEL: .visible .entry top_kernel(
#[no_mangle]
pub unsafe extern "ptx-kernel" fn top_kernel(a: *const u32, b: *mut u32) {
    // CHECK:      call.uni (retval0),
    // LLVM20-NEXT: [[IMPL_FN]]
    // LLVM21-SAME: [[IMPL_FN]]
    *b = deep::private::MyStruct::new(*a).square();
}

pub mod deep {
    pub mod private {
        pub struct MyStruct<T>(T);

        impl MyStruct<u32> {
            pub fn new(a: u32) -> Self {
                MyStruct(a)
            }

            #[inline(never)]
            pub fn square(&self) -> u32 {
                self.0.wrapping_mul(self.0)
            }
        }
    }
}
