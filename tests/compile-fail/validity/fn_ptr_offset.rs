use std::mem;

fn f() {}

fn main() {
    let x : fn() = f;
    let y : *mut u8 = unsafe { mem::transmute(x) };
    let y = y.wrapping_offset(1);
    let _x : fn() = unsafe { mem::transmute(y) }; //~ ERROR expected a function pointer
}
