// run-pass

#[allow(unreachable_code)]
fn main() {
    assert!(true, "Failed: {:?}", panic!("assert! evaluated format expressions"));
    debug_assert!(true, "Failed: {:?}", panic!("debug_assert! evaluated format expressions"));
    assert_eq!(1, 1, "Failed: {:?}", panic!("assert_eq! evaluated format expressions"));
    debug_assert_eq!(1, 1, "Failed: {:?}", panic!("debug_assert_eq! evaluated format expressions"));
    assert_ne!(1, 2, "Failed: {:?}", panic!("assert_ne! evaluated format expressions"));
    debug_assert_ne!(1, 2, "Failed: {:?}", panic!("debug_assert_ne! evaluated format expressions"));
}
