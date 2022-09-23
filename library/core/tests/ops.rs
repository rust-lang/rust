mod control_flow;
mod range;

use core::ops::{Deref, DerefMut};

// Test Deref implementations

#[test]
fn deref_mut_on_ref() {
    // Test that `&mut T` implements `DerefMut<T>`

    fn inc<T: Deref<Target = isize> + DerefMut>(mut t: T) {
        *t += 1;
    }

    let mut x: isize = 5;
    inc(&mut x);
    assert_eq!(x, 6);
}

#[test]
fn deref_on_ref() {
    // Test that `&T` and `&mut T` implement `Deref<T>`

    fn deref<U: Copy, T: Deref<Target = U>>(t: T) -> U {
        *t
    }

    let x: isize = 3;
    let y = deref(&x);
    assert_eq!(y, 3);

    let mut x: isize = 4;
    let y = deref(&mut x);
    assert_eq!(y, 4);
}

#[test]
#[allow(unreachable_code)]
fn test_not_never() {
    if !return () {}
}
