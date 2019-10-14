#![allow(dead_code)]

//! Used to test that certain lints don't trigger in imported external macros

#[macro_export]
macro_rules! foofoo {
    () => {
        loop {}
    };
}

#[macro_export]
macro_rules! must_use_unit {
    () => {
        #[must_use]
        fn foo() {}
    };
}
