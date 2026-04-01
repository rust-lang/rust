#![deny(rustdoc::private_doc_tests)]

mod foo {
    /// private doc test
    ///
    /// ```
    /// assert!(false);
    /// ```
    //~^^^^^ ERROR documentation test in private item
    pub fn bar() {}
}
