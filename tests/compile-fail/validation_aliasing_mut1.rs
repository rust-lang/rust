#![allow(unused_variables)]

mod safe {
    pub fn safe(x: &mut i32, y: &mut i32) {} //~ ERROR: in conflict with lock WriteLock
}

fn main() {
    let x = &mut 0 as *mut _;
    unsafe { safe::safe(&mut *x, &mut *x) };
}
