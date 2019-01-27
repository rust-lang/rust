#![no_std]
#![deny(warnings)]
#![feature(abi_ptx)]

// Verify the default CUDA arch.
// CHECK: .target sm_30
// CHECK: .address_size 64

// Verify function name doesn't contain unacceaptable characters.
// CHECK: .func (.param .b32 func_retval0) [[IMPL_FN:_ZN[a-zA-Z0-9$_]+square[a-zA-Z0-9$_]+]]

// CHECK-LABEL: .visible .entry top_kernel(
#[no_mangle]
pub unsafe extern "ptx-kernel" fn top_kernel(a: *const u32, b: *mut u32) {
    // CHECK:      call.uni (retval0),
    // CHECK-NEXT: [[IMPL_FN]]
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

// Verify that external function bodies are available.
// CHECK: .func (.param .b32 func_retval0) [[IMPL_FN]]
// CHECK: {
// CHECK:   mul.lo.s32 %{{r[0-9]+}}, %{{r[0-9]+}}, %{{r[0-9]+}}
// CHECK: }
