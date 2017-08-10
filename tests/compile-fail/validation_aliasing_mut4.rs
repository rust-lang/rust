#![allow(unused_variables)]

mod safe {
    use std::cell::Cell;

    // Make sure &mut UnsafeCell also has a lock to it
    pub fn safe(x: &mut Cell<i32>, y: &i32) {} //~ ERROR: in conflict with lock WriteLock
}

fn main() {
    let x = &mut 0 as *mut _;
    unsafe { safe::safe(&mut *(x as *mut _), &*x) };
}
