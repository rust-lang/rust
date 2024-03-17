// Check that `Option<u32>` in argument position and return position are represented as immediates
// with `ScalarPair(i1, i32)` "in-register" format.
//@ compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]

// CHECK: define void @arg(i1 noundef %_x.0, i32 %_x.1)
#[no_mangle]
pub fn arg(_x: Option<u32>) {}

// CHECK: define { i1, i32 } @ret()
#[no_mangle]
pub fn ret() -> Option<u32> {
    None
}
