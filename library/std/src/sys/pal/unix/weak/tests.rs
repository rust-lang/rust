use super::*;
use crate::ffi::c_int;

#[test]
fn dlsym() {
    // Try to find a symbol that definitely exists.
    dlsym! {
        fn abs(i: c_int) -> c_int;
    }

    let abs = abs.get().unwrap();
    assert_eq!(unsafe { abs(-1) }, 1);

    // Try to find a symbol that definitely does not exist.
    dlsym! {
        fn test_symbol_that_does_not_exist() -> c_int;
    }

    assert!(test_symbol_that_does_not_exist.get().is_none());
}
