//@ revisions: edition2015 edition2024
//@[edition2015] edition: 2015
//@[edition2024] edition: 2024
//@ aux-build:extern_macros.rs
//@ compile-flags:--test --test-arg=--test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
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
