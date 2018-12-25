// run-pass
// Tests that the result of type ascription has adjustments applied

#![feature(type_ascription)]

fn main() {
    let x = [1, 2, 3];
    // The RHS should coerce to &[i32]
    let _y : &[i32] = &x : &[i32; 3];
}
