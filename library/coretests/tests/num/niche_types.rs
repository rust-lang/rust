use core::num::NonZero;

#[test]
fn test_new_from_zero_is_none() {
    assert_eq!(NonZero::<char>::new(0 as char), None);
}

#[test]
fn test_new_from_extreme_is_some() {
    assert!(NonZero::<char>::new(1 as char).is_some());
    assert!(NonZero::<char>::new(char::MAX).is_some());
}
