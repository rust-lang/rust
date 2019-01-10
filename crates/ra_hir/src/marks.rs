//! This module implements manually tracked test coverage, which useful for
//! quickly finding a test responsible for testing a particular bit of code.
//!
//! See https://matklad.github.io/2018/06/18/a-trick-for-test-maintenance.html
//! for details, but the TL;DR is that you write your test as
//!
//! ```no-run
//! #[test]
//! fn test_foo() {
//!     covers!(test_foo);
//! }
//! ```
//!
//! and in the code under test you write
//!
//! ```no-run
//! fn foo() {
//!     if some_condition() {
//!         tested_by!(test_foo);
//!     }
//! }
//! ```
//!
//! This module then checks that executing the test indeed covers the specified
//! function. This is useful if you come back to the `foo` function ten years
//! later and wonder where the test are: now you can grep for `test_foo`.

#[macro_export]
macro_rules! tested_by {
    ($ident:ident) => {
        #[cfg(test)]
        {
            crate::marks::marks::$ident.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    };
}

#[macro_export]
macro_rules! covers {
    ($ident:ident) => {
        let _checker = crate::marks::marks::MarkChecker::new(&crate::marks::marks::$ident);
    };
}

#[cfg(test)]
pub(crate) mod marks {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(crate) struct MarkChecker {
        mark: &'static AtomicUsize,
        value_on_entry: usize,
    }

    impl MarkChecker {
        pub(crate) fn new(mark: &'static AtomicUsize) -> MarkChecker {
            let value_on_entry = mark.load(Ordering::SeqCst);
            MarkChecker {
                mark,
                value_on_entry,
            }
        }
    }

    impl Drop for MarkChecker {
        fn drop(&mut self) {
            if std::thread::panicking() {
                return;
            }
            let value_on_exit = self.mark.load(Ordering::SeqCst);
            assert!(value_on_exit > self.value_on_entry, "mark was not hit")
        }
    }

    macro_rules! mark {
        ($ident:ident) => {
            #[allow(bad_style)]
            pub(crate) static $ident: AtomicUsize = AtomicUsize::new(0);
        };
    }

    mark!(name_res_works_for_broken_modules);
}
