//@ compile-flags: -C no-prepopulate-passes

// Check that we use undef (and not zero) for uninitialized bytes in constants.

#![crate_type = "lib"]

use std::mem::MaybeUninit;

pub struct PartiallyUninit {
    x: u32,
    y: MaybeUninit<[u8; 10]>,
}

// CHECK: [[FULLY_UNINIT:@.*]] = private unnamed_addr constant [10 x i8] undef

// CHECK: [[PARTIALLY_UNINIT:@.*]] = private unnamed_addr constant <{ [4 x i8], [12 x i8] }> <{ [4 x i8] c"{{\\EF\\BE\\AD\\DE|\\DE\\AD\\BE\\EF}}", [12 x i8] undef }>, align 4

// This shouldn't contain undef, since it contains more chunks
// than the default value of uninit_const_chunk_threshold.
// CHECK: [[UNINIT_PADDING_HUGE:@.*]] = private unnamed_addr constant [32768 x i8] c"{{.+}}", align 4

// CHECK: [[FULLY_UNINIT_HUGE:@.*]] = private unnamed_addr constant [16384 x i8] undef

// CHECK-LABEL: @fully_uninit
#[no_mangle]
pub const fn fully_uninit() -> MaybeUninit<[u8; 10]> {
    const M: MaybeUninit<[u8; 10]> = MaybeUninit::uninit();
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 1 %_0, ptr align 1 {{.*}}[[FULLY_UNINIT]]{{.*}}, i{{(32|64)}} 10, i1 false)
    M
}

// CHECK-LABEL: @partially_uninit
#[no_mangle]
pub const fn partially_uninit() -> PartiallyUninit {
    const X: PartiallyUninit = PartiallyUninit { x: 0xdeadbeef, y: MaybeUninit::uninit() };
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 4 %_0, ptr align 4 {{.*}}[[PARTIALLY_UNINIT]]{{.*}}, i{{(32|64)}} 16, i1 false)
    X
}

// CHECK-LABEL: @uninit_padding_huge
#[no_mangle]
pub const fn uninit_padding_huge() -> [(u32, u8); 4096] {
    const X: [(u32, u8); 4096] = [(123, 45); 4096];
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 4 %_0, ptr align 4 {{.*}}[[UNINIT_PADDING_HUGE]]{{.*}}, i{{(32|64)}} 32768, i1 false)
    X
}

// CHECK-LABEL: @fully_uninit_huge
#[no_mangle]
pub const fn fully_uninit_huge() -> MaybeUninit<[u32; 4096]> {
    const F: MaybeUninit<[u32; 4096]> = MaybeUninit::uninit();
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 4 %_0, ptr align 4 {{.*}}[[FULLY_UNINIT_HUGE]]{{.*}}, i{{(32|64)}} 16384, i1 false)
    F
}
