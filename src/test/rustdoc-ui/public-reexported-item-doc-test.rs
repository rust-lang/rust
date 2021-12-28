// check-pass

#![deny(rustdoc::private_doc_tests)]

mod foo {
    /// re-exported doc test
    ///
    /// ```
    /// assert!(true);
    /// ```
    pub fn bar() {}
}

pub use foo::bar;
