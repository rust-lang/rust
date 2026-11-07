//@ aux-build: nounwind_inference_aux.rs
//@ needs-unwind
//@ compile-flags: -C no-prepopulate-passes
//@ revisions: SIZE SIZEMIN
//@[SIZE] compile-flags: -Copt-level=s
//@[SIZEMIN] compile-flags: -Copt-level=z

#![crate_type = "lib"]

extern crate nounwind_inference_aux;

struct S;
impl Drop for S {
    #[inline(never)]
    fn drop(&mut self) {}
}

// CHECK-DAG: call noundef i32 @aux_quiet(i32 noundef %x) [[QUIET_CALL:#[0-9]+]]
// CHECK-DAG: attributes [[QUIET_CALL]] = { {{[^}]*}}nounwind
// CHECK-DAG: invoke noundef i32 @aux_panicky(i32 noundef %x)
#[no_mangle]
pub fn use_aux_quiet(x: u32) -> u32 {
    let _s = S;
    nounwind_inference_aux::aux_quiet(x)
}

#[no_mangle]
pub fn use_aux_panicky(x: u32) -> u32 {
    let _s = S;
    nounwind_inference_aux::aux_panicky(x)
}
