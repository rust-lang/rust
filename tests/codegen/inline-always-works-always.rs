//@ revisions: NO-OPT SIZE-OPT SPEED-OPT
//@[NO-OPT] compile-flags: -Copt-level=0
//@[SIZE-OPT] compile-flags: -Copt-level=s
//@[SPEED-OPT] compile-flags: -Copt-level=3

#![crate_type = "rlib"]

#[no_mangle]
#[inline(always)]
pub extern "C" fn callee() -> u32 {
    4 + 4
}

// CHECK-LABEL: caller
// SIZE-OPT: ret i32 8
// SPEED-OPT: ret i32 8
// NO-OPT: ret i32 8
#[no_mangle]
pub extern "C" fn caller() -> u32 {
    callee()
}
