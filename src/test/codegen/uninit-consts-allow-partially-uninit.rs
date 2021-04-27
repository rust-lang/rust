// compile-flags: -C no-prepopulate-passes -Z partially_uninit_const_threshold=1024

// Like uninit-consts.rs, but tests that we correctly generate partially-uninit consts
// when the (disabled by default) partially_uninit_const_threshold flag is used.

#![crate_type = "lib"]

use std::mem::MaybeUninit;

pub struct PartiallyUninit {
    x: u32,
    y: MaybeUninit<[u8; 10]>
}

// This should be partially undef.
// CHECK: [[PARTIALLY_UNINIT:@[0-9]+]] = private unnamed_addr constant <{ [4 x i8], [12 x i8] }> <{ [4 x i8] c"\EF\BE\AD\DE", [12 x i8] undef }>, align 4

// This shouldn't contain undef, since it's larger than the 1024 byte limit.
// CHECK: [[UNINIT_PADDING_HUGE:@[0-9]+]] = private unnamed_addr constant <{ [32768 x i8] }> <{ [32768 x i8] c"{{.+}}" }>, align 4

// CHECK-LABEL: @partially_uninit
#[no_mangle]
pub const fn partially_uninit() -> PartiallyUninit {
    const X: PartiallyUninit = PartiallyUninit { x: 0xdeadbeef, y: MaybeUninit::uninit() };
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i{{(32|64)}}(i8* align 4 %1, i8* align 4 getelementptr inbounds (<{ [4 x i8], [12 x i8] }>, <{ [4 x i8], [12 x i8] }>* [[PARTIALLY_UNINIT]], i32 0, i32 0, i32 0), i{{(32|64)}} 16, i1 false)
    X
}

// CHECK-LABEL: @uninit_padding_huge
#[no_mangle]
pub const fn uninit_padding_huge() -> [(u32, u8); 4096] {
    const X: [(u32, u8); 4096] = [(123, 45); 4096];
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i{{(32|64)}}(i8* align 4 %1, i8* align 4 getelementptr inbounds (<{ [32768 x i8] }>, <{ [32768 x i8] }>* [[UNINIT_PADDING_HUGE]], i32 0, i32 0, i32 0), i{{(32|64)}} 32768, i1 false)
    X
}
