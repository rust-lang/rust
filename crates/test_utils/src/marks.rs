//! This module implements manually tracked test coverage, which useful for
//! quickly finding a test responsible for testing a particular bit of code.
//!
//! See <https://matklad.github.io/2018/06/18/a-trick-for-test-maintenance.html>
//! for details, but the TL;DR is that you write your test as
//!
//! ```
//! #[test]
//! fn test_foo() {
//!     covers!(test_foo);
//! }
//! ```
//!
//! and in the code under test you write
//!
//! ```
//! # use test_utils::tested_by;
//! # fn some_condition() -> bool { true }
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
use std::sync::atomic::{AtomicUsize, Ordering};

#[macro_export]
macro_rules! tested_by {
    ($ident:ident) => {{
        #[cfg(test)]
        {
            // sic! use call-site crate
            crate::marks::$ident.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }};
}

#[macro_export]
macro_rules! covers {
    ($ident:ident) => {
        // sic! use call-site crate
        let _checker = $crate::marks::MarkChecker::new(&crate::marks::$ident);
    };
}

#[macro_export]
macro_rules! marks {
    ($($ident:ident)*) => {
        $(
        #[allow(bad_style)]
        pub(crate) static $ident: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        )*
    };
}

pub struct MarkChecker {
    mark: &'static AtomicUsize,
    value_on_entry: usize,
}

impl MarkChecker {
    pub fn new(mark: &'static AtomicUsize) -> MarkChecker {
        let value_on_entry = mark.load(Ordering::SeqCst);
        MarkChecker { mark, value_on_entry }
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
