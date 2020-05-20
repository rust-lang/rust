//! This module implements manually tracked test coverage, which is useful for
//! quickly finding a test responsible for testing a particular bit of code.
//!
//! See <https://matklad.github.io/2018/06/18/a-trick-for-test-maintenance.html>
//! for details, but the TL;DR is that you write your test as
//!
//! ```
//! #[test]
//! fn test_foo() {
//!     mark::check!(test_foo);
//! }
//! ```
//!
//! and in the code under test you write
//!
//! ```
//! # use test_utils::mark;
//! # fn some_condition() -> bool { true }
//! fn foo() {
//!     if some_condition() {
//!         mark::hit!(test_foo);
//!     }
//! }
//! ```
//!
//! This module then checks that executing the test indeed covers the specified
//! function. This is useful if you come back to the `foo` function ten years
//! later and wonder where the test are: now you can grep for `test_foo`.
use std::sync::atomic::{AtomicUsize, Ordering};

#[macro_export]
macro_rules! _hit {
    ($ident:ident) => {{
        #[cfg(test)]
        {
            extern "C" {
                #[no_mangle]
                static $ident: std::sync::atomic::AtomicUsize;
            }
            unsafe {
                $ident.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
    }};
}
pub use _hit as hit;

#[macro_export]
macro_rules! _check {
    ($ident:ident) => {
        #[no_mangle]
        static $ident: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let _checker = $crate::mark::MarkChecker::new(&$ident);
    };
}
pub use _check as check;

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
