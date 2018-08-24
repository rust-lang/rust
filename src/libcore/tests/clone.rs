#[test]
fn test_borrowed_clone() {
    let x = 5;
    let y: &i32 = &x;
    let z: &i32 = (&y).clone();
    assert_eq!(*z, 5);
}

#[test]
fn test_clone_from() {
    let a = box 5;
    let mut b = box 10;
    b.clone_from(&a);
    assert_eq!(*b, 5);
}
