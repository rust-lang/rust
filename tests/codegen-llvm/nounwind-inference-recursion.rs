//@ needs-unwind
//@ compile-flags: -C no-prepopulate-passes
//@ revisions: SIZE SIZEMIN
//@[SIZE] compile-flags: -Copt-level=s
//@[SIZEMIN] compile-flags: -Copt-level=z

// A call back into the body being analyzed counts as unable to unwind, so a recursive function
// whose other operations are quiet is still `nounwind`.
//
// The callers hold a live `Drop` local, which forces an `invoke` unless the callee is proven not
// to unwind.
//
// CHECK: define {{.*}} @{{.*}}self_quiet(i32 noundef %n) unnamed_addr [[QUIET_DEF:#[0-9]+]]
// CHECK: call noundef i32 @{{.*}}self_quiet(i32 noundef %_2) [[QUIET_CALL:#[0-9]+]]
// CHECK: define {{.*}} @{{.*}}self_panicky(i32 noundef %n) unnamed_addr
//
// The same shape with one panicking operation keeps its `invoke`, so the checks above cannot pass
// vacuously.
// CHECK-DAG: call noundef i32 @{{.*}}self_quiet(i32 noundef %x) [[QUIET_CALL]]
// CHECK-DAG: invoke noundef i32 @{{.*}}self_panicky(i32 noundef %x)
//
// CHECK-DAG: attributes [[QUIET_DEF]] = { {{[^}]*}}nounwind{{[^}]*}} }
// CHECK-DAG: attributes [[QUIET_CALL]] = { {{[^}]*}}nounwind{{[^}]*}} }
#![crate_type = "lib"]

struct Guard;

impl Drop for Guard {
    #[inline(never)]
    fn drop(&mut self) {}
}

#[inline(never)]
pub fn self_quiet(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    self_quiet(n - 1)
}

#[inline(never)]
pub fn self_panicky(n: u32) -> u32 {
    if n == 0 {
        panic!("bottom");
    }
    self_panicky(n - 1)
}

#[no_mangle]
pub fn use_self_quiet(x: u32) -> u32 {
    let _g = Guard;
    self_quiet(x)
}

#[no_mangle]
pub fn use_self_panicky(x: u32) -> u32 {
    let _g = Guard;
    self_panicky(x)
}
