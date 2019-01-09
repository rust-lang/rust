#![deny(private_doc_tests)]

mod foo {
    /// private doc test
    ///
    /// ```
    /// assert!(false);
    /// ```
    //~^^^^^ ERROR Documentation test in private item
    fn bar() {}
}
