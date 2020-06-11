// compile-flags:--test --test-args=--test-threads=1
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// check-pass

//! ```
//! assert_eq!(1 + 1, 2);
//! ```

pub mod foo {

    /// ```
    /// assert_eq!(1 + 1, 2);
    /// ```
    pub fn bar() {}
}
