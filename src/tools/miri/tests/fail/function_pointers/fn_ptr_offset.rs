// Validation makes this fail in the wrong place
//@compile-flags: -Zmiri-disable-validation

use std::mem;

fn f() {}

fn main() {
    let x: fn() = f;
    let y: *mut u8 = unsafe { mem::transmute(x) };
    let y = y.wrapping_offset(1);
    let x: fn() = unsafe { mem::transmute(y) };
    x(); //~ ERROR: function pointer but it does not point to a function
}
