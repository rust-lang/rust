//@ revisions: DEBUGINFO NODEBUGINFO
//@ compile-flags: -Copt-level=3 -Cno-prepopulate-passes
//@ [DEBUGINFO] compile-flags: -Cdebuginfo=full

// From https://github.com/rust-lang/rust/issues/128081.
// Ensure that we only generate a memcpy instruction.

#![crate_type = "lib"]

#[derive(Clone)]
struct SubCloneAndCopy {
    v1: u32,
    v2: u32,
}

#[derive(Clone)]
struct CloneOnly {
    v1: u8,
    v2: u8,
    v3: u8,
    v4: u8,
    v5: u8,
    v6: u8,
    v7: u8,
    v8: u8,
    v9: u8,
    v_sub: SubCloneAndCopy,
    v_large: [u8; 256],
}

// CHECK-LABEL: define {{.*}}@clone_only(
#[no_mangle]
pub fn clone_only(v: &CloneOnly) -> CloneOnly {
    // CHECK-NOT: call {{.*}}clone
    // CHECK-NOT: store i8
    // CHECK-NOT: store i32
    // CHECK: call void @llvm.memcpy
    // CHECK-NEXT: ret void
    v.clone()
}
