//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// Non-overlapping scopes should produce correct llvm.lifetimes,
// which allow reuse of the same stack allocation.

pub struct WithOffset<T> {
    pub data: T,
    pub offset: usize,
}

#[inline(never)]
pub fn peak_w(w: &WithOffset<&[u8; 16]>) {
    std::hint::black_box(w);
}

#[inline(never)]
pub fn use_w(w: WithOffset<&[u8; 16]>) {
    std::hint::black_box(w);
}

// CHECK-LABEL: define void @scoped_small_structs(
// CHECK-NEXT: start:
// CHECK-NEXT: [[B:%.*]] = alloca
// CHECK-NEXT: [[A:%.*]] = alloca
// CHECK: call void @llvm.lifetime.start.p0(i64 16, ptr nonnull [[A]])
// CHECK: call void @llvm.lifetime.end.p0(i64 16, ptr nonnull [[A]])
// CHECK: call void @llvm.lifetime.start.p0(i64 16, ptr nonnull [[B]])
// CHECK: call void @llvm.lifetime.end.p0(i64 16, ptr nonnull [[B]])
#[no_mangle]
pub fn scoped_small_structs(buf: [u8; 16]) {
    {
        let w = WithOffset { data: &buf, offset: 0 };

        peak_w(&w);
        use_w(w);
    }
    {
        let w2 = WithOffset { data: &buf, offset: 1 };

        peak_w(&w2);
        use_w(w2);
    }
}
