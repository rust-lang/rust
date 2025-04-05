//@ compile-flags: -C opt-level=3
//! Ensure that matching on `x % 5` generates an unreachable
//! branch for values greater than 4.
//! Discovered in issue #93514.

#![crate_type = "lib"]

#[no_mangle]
pub unsafe fn parse0(x: u32) -> u32 {
    // CHECK-LABEL: i32 @parse0(
    // CHECK:    [[_2:%.*]] = urem
    // CHECK-NEXT:    switch i32 [[_2]], label %[[DEFAULT_UNREACHABLE1:.*]] [
    // CHECK-NEXT:      i32 0
    // CHECK-NEXT:      i32 1
    // CHECK-NEXT:      i32 2
    // CHECK-NEXT:      i32 3
    // CHECK-NEXT:      i32 4
    // CHECK-NEXT:    ]
    // CHECK:       [[DEFAULT_UNREACHABLE1]]:
    // CHECK-NEXT:    unreachable
    // CHECK:    ret i32
    match x % 5 {
        0 => f1(x),
        1 => f2(x),
        2 => f3(x),
        3 => f4(x),
        4 => f5(x),
        _ => eliminate_me(),
    }
}

extern "Rust" {
    fn eliminate_me() -> u32;
    fn f1(x: u32) -> u32;
    fn f2(x: u32) -> u32;
    fn f3(x: u32) -> u32;
    fn f4(x: u32) -> u32;
    fn f5(x: u32) -> u32;
}
