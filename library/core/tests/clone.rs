#[test]
#[allow(suspicious_double_ref_op)]
fn test_borrowed_clone() {
    let x = 5;
    let y: &i32 = &x;
    let z: &i32 = (&y).clone();
    assert_eq!(*z, 5);
}

#[test]
fn test_clone_from() {
    let a = Box::new(5);
    let mut b = Box::new(10);
    b.clone_from(&a);
    assert_eq!(*b, 5);
}
