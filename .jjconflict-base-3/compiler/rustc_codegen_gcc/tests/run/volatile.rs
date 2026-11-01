// Compiler:
//
// Run-time:
//   status: 0

use std::mem::MaybeUninit;

#[allow(dead_code)]
#[derive(Debug)]
struct Struct {
    pointer: *const (),
    func: unsafe fn(*const ()),
}

fn func(_ptr: *const ()) {}

fn main() {
    let mut x = MaybeUninit::<&Struct>::uninit();
    x.write(&Struct { pointer: std::ptr::null(), func });
    let x = unsafe { x.assume_init() };
    let value = unsafe { (x as *const Struct).read_volatile() };
    println!("{:?}", value);
}
