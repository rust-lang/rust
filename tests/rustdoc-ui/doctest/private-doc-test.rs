//@ check-pass

#![deny(rustdoc::private_doc_tests)]

mod foo {
    /// private doc test
    ///
    /// ```ignore (used for testing ignored doc tests)
    /// assert!(false);
    /// ```
    fn bar() {}
}
