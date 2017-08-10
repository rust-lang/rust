#![allow(unused_variables)]

#[repr(u32)]
enum Bool { True }

mod safe {
    pub(crate) fn safe(x: &mut super::Bool) {
        let x = x as *mut _ as *mut u32;
        unsafe { *x = 44; } // out-of-bounds enum discriminant
    }
}

fn main() {
    let mut x = Bool::True;
    safe::safe(&mut x); //~ ERROR: invalid enum discriminant
}
