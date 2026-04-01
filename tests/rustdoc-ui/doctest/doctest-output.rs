//@ revisions: edition2015 edition2024
//@[edition2015]edition:2015
//@[edition2015]aux-build:extern_macros.rs
//@[edition2015]compile-flags:--test --test-args=--test-threads=1
//@[edition2024]edition:2024
//@[edition2024]aux-build:extern_macros.rs
//@[edition2024]compile-flags:--test --test-args=--test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "ran in \d+\.\d+s" -> "ran in $$TIME"
//@ normalize-stdout: "compilation took \d+\.\d+s" -> "compilation took $$TIME"
//@ check-pass

//! ```
//! assert_eq!(1 + 1, 2);
//! ```

extern crate extern_macros as macros;

use macros::attrs_on_struct;

pub mod foo {

    /// ```
    /// assert_eq!(1 + 1, 2);
    /// ```
    pub fn bar() {}
}

attrs_on_struct! {
    /// ```
    /// assert!(true);
    /// ```
}
