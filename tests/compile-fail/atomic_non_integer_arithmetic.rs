#![feature(core_intrinsics)]

pub fn main() {
    let mut z: f64 = 1.0;
    unsafe {
        ::std::intrinsics::atomic_xadd(&mut z, 2.0);
        //~^ ERROR: Atomic arithmetic operations only work on integer types
    }
}
