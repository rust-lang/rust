#![allow(unused_variables)]

static mut PTR: *mut u8 = 0 as *mut _;

fn fun1(x: &mut u8) {
    unsafe {
        PTR = x;
    }
}

fn fun2() {
    // Now we use a pointer we are not allowed to use
    let _x = unsafe { *PTR }; //~ ERROR: in conflict with lock WriteLock
}

fn main() {
    let mut val = 0;
    fun1(&mut val);
    fun2();
}
