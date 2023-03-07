#![allow(unused)]
#![warn(clippy::size_of_ref)]

use std::mem::size_of_val;

fn main() {
    let x = 5;
    let y = &x;

    size_of_val(&x); // no lint
    size_of_val(y); // no lint

    size_of_val(&&x);
    size_of_val(&y);
}

struct S {
    field: u32,
    data: Vec<u8>,
}

impl S {
    /// Get size of object including `self`, in bytes.
    pub fn size(&self) -> usize {
        std::mem::size_of_val(&self) + (std::mem::size_of::<u8>() * self.data.capacity())
    }
}
