//@ revisions: no-opt size-opt speed-opt
//@[no-opt] compile-flags: -Copt-level=0
//@[size-opt] compile-flags: -Copt-level=s
//@[speed-opt] compile-flags: -Copt-level=3

#![crate_type = "rlib"]

#[no_mangle]
#[inline(always)]
pub extern "C" fn callee() -> u32 {
    4 + 4
}

// CHECK-LABEL: caller
// CHECK-SIZE-OPT: ret i32 8
// CHECK-SPEED-OPT: ret i32 8
// CHECK-NO-OPT: ret i32 8
#[no_mangle]
pub extern "C" fn caller() -> u32 {
    callee()
}
