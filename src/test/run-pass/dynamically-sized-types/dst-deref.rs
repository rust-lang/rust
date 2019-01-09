// run-pass
// Test that a custom deref with a fat pointer return type does not ICE


use std::ops::Deref;

pub struct Arr {
    ptr: Box<[usize]>
}

impl Deref for Arr {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        &*self.ptr
    }
}

pub fn foo(arr: &Arr) {
    assert_eq!(arr.len(), 3);
    let x: &[usize] = &**arr;
    assert_eq!(x[0], 1);
    assert_eq!(x[1], 2);
    assert_eq!(x[2], 3);
}

fn main() {
    let a = Arr { ptr: Box::new([1, 2, 3]) };
    foo(&a);
}
