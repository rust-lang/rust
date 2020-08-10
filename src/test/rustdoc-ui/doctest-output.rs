// edition:2018
// aux-build:extern_macros.rs
// compile-flags:--test --test-args=--test-threads=1
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// check-pass

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
