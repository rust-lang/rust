use super::*;
use crate::ffi::c_int;

#[test]
fn dlsym_existing() {
    // Try to find a symbol that definitely exists.
    dlsym! {
        fn abs(i: c_int) -> c_int;
    }

    dlsym! {
        #[link_name = "abs"]
        fn custom_name(i: c_int) -> c_int;
    }

    let abs = abs.get().unwrap();
    assert_eq!(unsafe { abs(-1) }, 1);

    let custom_name = custom_name.get().unwrap();
    assert_eq!(unsafe { custom_name(-1) }, 1);
}

#[test]
fn dlsym_missing() {
    // Try to find a symbol that definitely does not exist.
    dlsym! {
        fn test_symbol_that_does_not_exist() -> c_int;
    }

    assert!(test_symbol_that_does_not_exist.get().is_none());
}
