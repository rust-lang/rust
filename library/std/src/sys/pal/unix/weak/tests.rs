use super::*;

#[test]
fn dlsym_existing() {
    const TEST_STRING: &'static CStr = c"Ferris!";

    // Try to find a symbol that definitely exists.
    dlsym! {
        fn strlen(cs: *const c_char) -> usize;
    }

    dlsym! {
        #[link_name = "strlen"]
        fn custom_name(cs: *const c_char) -> usize;
    }

    let strlen = strlen.get().unwrap();
    assert_eq!(unsafe { strlen(TEST_STRING.as_ptr()) }, TEST_STRING.count_bytes());

    let custom_name = custom_name.get().unwrap();
    assert_eq!(unsafe { custom_name(TEST_STRING.as_ptr()) }, TEST_STRING.count_bytes());
}

#[test]
fn dlsym_missing() {
    // Try to find a symbol that definitely does not exist.
    dlsym! {
        fn test_symbol_that_does_not_exist() -> i32;
    }

    assert!(test_symbol_that_does_not_exist.get().is_none());
}
