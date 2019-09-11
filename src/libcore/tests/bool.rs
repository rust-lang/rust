#[test]
fn test_bool_to_option() {
    assert_eq!(false.then(0), None);
    assert_eq!(true.then(0), Some(0));
    assert_eq!(false.then_with(|| 0), None);
    assert_eq!(true.then_with(|| 0), Some(0));
}
