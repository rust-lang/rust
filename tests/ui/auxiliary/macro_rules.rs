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

#[macro_export]
macro_rules! try_err {
    () => {
        pub fn try_err_fn() -> Result<i32, i32> {
            let err: i32 = 1;
            // To avoid warnings during rustfix
            if true {
                Err(err)?
            } else {
                Ok(2)
            }
        }
    };
}
