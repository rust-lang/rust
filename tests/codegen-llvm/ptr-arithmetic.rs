//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled

#![crate_type = "lib"]

// CHECK-LABEL: ptr @i32_add(
// CHECK-SAME: [[WORD:i[0-9]+]] noundef %n)
#[no_mangle]
pub unsafe fn i32_add(p: *const i32, n: usize) -> *const i32 {
    // CHECK: %[[TEMP:.+]] = getelementptr inbounds{{( nuw)?}} i32, ptr %p, [[WORD]] %n
    // CHECK: ret ptr %[[TEMP]]
    p.add(n)
}

// Ensure we tell LLVM that the negation in `sub` can't overflow.

// CHECK-LABEL: ptr @i32_sub(
// CHECK-SAME: [[WORD:i[0-9]+]] noundef %n)
#[no_mangle]
pub unsafe fn i32_sub(p: *const i32, n: usize) -> *const i32 {
    // CHECK: %[[DELTA:.+]] = sub nsw [[WORD]] 0, %n
    // CHECK: %[[TEMP:.+]] = getelementptr inbounds i32, ptr %p, [[WORD]] %[[DELTA]]
    // CHECK: ret ptr %[[TEMP]]
    p.sub(n)
}

// CHECK-LABEL: ptr @i32_offset(
// CHECK-SAME: [[WORD:i[0-9]+]] noundef %d)
#[no_mangle]
pub unsafe fn i32_offset(p: *const i32, d: isize) -> *const i32 {
    // CHECK: %[[TEMP:.+]] = getelementptr inbounds i32, ptr %p, [[WORD]] %d
    // CHECK: ret ptr %[[TEMP]]
    p.offset(d)
}
