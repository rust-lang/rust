#[test]
fn test_bool_to_option() {
    assert_eq!(false.to_option(0), None);
    assert_eq!(true.to_option(0), Some(0));
    assert_eq!(false.to_option_with(|| 0), None);
    assert_eq!(true.to_option_with(|| 0), Some(0));
}
