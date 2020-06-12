#[test]
fn test_bool_to_option() {
    assert_eq!(false.then_some(0), None);
    assert_eq!(true.then_some(0), Some(0));
    assert_eq!(false.then(|| 0), None);
    assert_eq!(true.then(|| 0), Some(0));
}
