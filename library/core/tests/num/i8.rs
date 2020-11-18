int_module!(i8, i8);

#[test]
fn test_incr_operation() {
    let mut x: i8 = -12;
    let y: i8 = -12;
    x = x + 1;
    x = x - 1;
    assert_eq!(x, y);
}
