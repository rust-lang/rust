#![allow(unused_variables)]

mod safe {
    pub fn safe(x: &i32, y: &mut i32) {} //~ ERROR: in conflict with lock ReadLock
}

fn main() {
    let x = &mut 0 as *mut _;
    unsafe { safe::safe(&*x, &mut *x) };
}
