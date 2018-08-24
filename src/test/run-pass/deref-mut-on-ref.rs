// Test that `&mut T` implements `DerefMut<T>`


use std::ops::{Deref, DerefMut};

fn inc<T: Deref<Target=isize> + DerefMut>(mut t: T) {
    *t += 1;
}

fn main() {
    let mut x: isize = 5;
    inc(&mut x);
    assert_eq!(x, 6);
}
