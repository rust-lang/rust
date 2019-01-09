// Test that `&T` and `&mut T` implement `Deref<T>`


use std::ops::Deref;

fn deref<U:Copy,T:Deref<Target=U>>(t: T) -> U {
    *t
}

fn main() {
    let x: isize = 3;
    let y = deref(&x);
    assert_eq!(y, 3);

    let mut x: isize = 4;
    let y = deref(&mut x);
    assert_eq!(y, 4);
}
