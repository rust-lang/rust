// This file is included by both implementations of `weak!`.
use super::weak;
use crate::ffi::{CStr, c_char};

#[test]
fn weak_existing() {
    const TEST_STRING: &'static CStr = c"Ferris!";

    // Try to find a symbol that definitely exists.
    weak! {
        fn strlen(cs: *const c_char) -> usize;
    }

    let strlen = strlen.get().unwrap();
    assert_eq!(unsafe { strlen(TEST_STRING.as_ptr()) }, TEST_STRING.count_bytes());
}

#[test]
fn weak_missing() {
    // Try to find a symbol that definitely does not exist.
    weak! {
        fn test_symbol_that_does_not_exist() -> i32;
    }

    assert!(test_symbol_that_does_not_exist.get().is_none());
}
