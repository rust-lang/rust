use std::mem;

fn f() {}

fn main() {
    let x : fn() = f;
    let y : *mut u8 = unsafe { mem::transmute(x) };
    let y = y.wrapping_offset(1);
    let x : fn() = unsafe { mem::transmute(y) };
    x(); //~ ERROR: tried to use an integer pointer or a dangling pointer as a function pointer
}
