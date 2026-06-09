//@ check-pass

#![deny(rustdoc::private_doc_tests)]

pub fn foo() {}

mod private {
    /// re-exported doc test
    ///
    /// ```
    /// assert!(true);
    /// ```
    pub fn bar() {}
}

pub use private::bar;
