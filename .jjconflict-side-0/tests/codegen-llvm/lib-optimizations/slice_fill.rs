//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

use std::mem::MaybeUninit;

// CHECK-LABEL: @slice_fill_pass_undef
#[no_mangle]
pub fn slice_fill_pass_undef(s: &mut [MaybeUninit<u8>], v: MaybeUninit<u8>) {
    // CHECK: tail call void @llvm.memset.{{.*}}(ptr nonnull align 1 %s.0, i8 %v, {{.*}} %s.1, i1 false)
    // CHECK: ret
    s.fill(v);
}

// CHECK-LABEL: @slice_fill_uninit
#[no_mangle]
pub fn slice_fill_uninit(s: &mut [MaybeUninit<u8>]) {
    // CHECK-NOT: call
    // CHECK: ret void
    s.fill(MaybeUninit::uninit());
}

// CHECK-LABEL: @slice_wide_memset
#[no_mangle]
pub fn slice_wide_memset(s: &mut [u16]) {
    // CHECK: tail call void @llvm.memset.{{.*}}(ptr nonnull align 2 %s.0, i8 -1
    // CHECK: ret
    s.fill(0xFFFF);
}
