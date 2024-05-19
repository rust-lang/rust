// Compiler:
//
// Run-time:
//   status: 0

use std::mem::MaybeUninit;

#[derive(Debug)]
struct Struct {
    pointer: *const (),
    func: unsafe fn(*const ()),
}

fn func(ptr: *const ()) {
}

fn main() {
    let mut x = MaybeUninit::<&Struct>::uninit();
    x.write(&Struct {
        pointer: std::ptr::null(),
        func,
    });
    let x = unsafe { x.assume_init() };
    let value = unsafe { (x as *const Struct).read_volatile() };
    println!("{:?}", value);
}
