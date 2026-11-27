//@ needs-unwind
//@ compile-flags: -C no-prepopulate-passes -Copt-level=s -Znounwind-inference

// A body that provably never unwinds is `nounwind`, and calls to it are plain `call`s even with a
// live `Drop` local, which otherwise forces an `invoke`.
//
// CHECK-DAG: define {{.*}} @quiet(i32 noundef %x) unnamed_addr [[QUIET_DEF:#[0-9]+]]
// CHECK-DAG: attributes [[QUIET_DEF]] = { {{[^}]*}}nounwind{{[^}]*}} }
// CHECK-DAG: call noundef i32 @quiet(i32 noundef %x) [[QUIET_CALL:#[0-9]+]]
// CHECK-DAG: attributes [[QUIET_CALL]] = { {{[^}]*}}nounwind{{[^}]*}} }
// The can-unwind case keeps its `invoke`.
// CHECK-DAG: invoke noundef i32 @panicky(i32 noundef %x)
#![crate_type = "lib"]

struct S;

impl Drop for S {
    #[inline(never)]
    fn drop(&mut self) {}
}

#[inline(never)]
#[no_mangle]
pub fn quiet(x: u32) -> u32 {
    x.wrapping_add(1)
}

#[inline(never)]
#[no_mangle]
pub fn panicky(x: u32) -> u32 {
    if x == 0 {
        panic!("zero");
    }
    x
}

#[no_mangle]
pub fn use_quiet(x: u32) -> u32 {
    let _s = S;
    quiet(x)
}

#[no_mangle]
pub fn use_panicky(x: u32) -> u32 {
    let _s = S;
    panicky(x)
}
