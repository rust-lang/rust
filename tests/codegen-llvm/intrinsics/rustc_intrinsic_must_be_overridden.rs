//@ revisions: OPT0 OPT1
//@ [OPT0] compile-flags: -Copt-level=0
//@ [OPT1] compile-flags: -Copt-level=1
//@ compile-flags: -Cno-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// CHECK-NOT: core::intrinsics::size_of_val

#[no_mangle]
pub unsafe fn size_of_val(ptr: *const i32) -> usize {
    core::intrinsics::size_of_val(ptr)
}
