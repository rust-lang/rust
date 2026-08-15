//@ run-pass
// Test that a custom deref with a fat pointer return type does not ICE


use std::ops::{Deref, DerefMut};

pub struct Arr {
    ptr: Box<[usize]>
}

impl Deref for Arr {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        panic!();
    }
}

impl DerefMut for Arr {
    fn deref_mut(&mut self) -> &mut [usize] {
        &mut *self.ptr
    }
}

pub fn foo(arr: &mut Arr) {
    let x: &mut [usize] = &mut **arr;
    assert_eq!(x[0], 1);
    assert_eq!(x[1], 2);
    assert_eq!(x[2], 3);
}

fn main() {
    let mut a = Arr { ptr: Box::new([1, 2, 3]) };
    foo(&mut a);
}
