// run-pass

pub fn main() {
    let mut a: Vec<Box<_>> = vec![Box::new(10)];
    let b = a.clone();

    assert_eq!(*a[0], 10);
    assert_eq!(*b[0], 10);

    // This should only modify the value in a, not b
    *a[0] = 20;

    assert_eq!(*a[0], 20);
    assert_eq!(*b[0], 10);
}
