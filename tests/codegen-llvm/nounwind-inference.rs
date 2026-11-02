//@ needs-unwind
//@ compile-flags: -C no-prepopulate-passes
//@ revisions: SIZE SIZEMIN
//@[SIZE] compile-flags: -Copt-level=s
//@[SIZEMIN] compile-flags: -Copt-level=z

// A function whose body provably never unwinds is marked `nounwind`, and calls to it are plain
// `call`s even in a function with a live `Drop` local, which otherwise forces an `invoke`.
//
// CHECK-DAG: define {{.*}} @quiet(i32 noundef %x) unnamed_addr [[QUIET_DEF:#[0-9]+]]
// CHECK-DAG: attributes [[QUIET_DEF]] = { {{[^}]*}}nounwind{{.*}} }
// CHECK-DAG: call noundef i32 @quiet(i32 noundef %x) [[QUIET_CALL:#[0-9]+]]
// CHECK-DAG: attributes [[QUIET_CALL]] = { {{[^}]*}}nounwind{{[^}]*}} }
//
// The same shape with a call that really can unwind keeps its `invoke`, so the checks above
// cannot pass vacuously.
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
