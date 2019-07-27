// run-pass
#![feature(extern_types)]

use std::mem::{size_of_val, align_of_val};

extern {
    type A;
}

fn main() {
    let x: &A = unsafe {
        &*(1usize as *const A)
    };

    assert_eq!(size_of_val(x), 0);
    assert_eq!(align_of_val(x), 1);
}
