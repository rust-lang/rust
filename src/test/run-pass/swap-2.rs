use std::mem::swap;

pub fn main() {
    let mut a: Vec<isize> = vec![0, 1, 2, 3, 4, 5, 6];
    a.swap(2, 4);
    assert_eq!(a[2], 4);
    assert_eq!(a[4], 2);
    let mut n = 42;
    swap(&mut n, &mut a[0]);
    assert_eq!(a[0], 42);
    assert_eq!(n, 0);
}
