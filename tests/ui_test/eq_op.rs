#[warn(clippy::eq_op)]
#[test]
fn eq_op_shouldnt_trigger_in_tests() {
    let a = 1;
    let result = a + 1 == 1 + a;
    assert!(result);
}

#[test]
fn eq_op_macros_shouldnt_trigger_in_tests() {
    let a = 1;
    let b = 2;
    assert_eq!(a, a);
    assert_eq!(a + b, b + a);
}
