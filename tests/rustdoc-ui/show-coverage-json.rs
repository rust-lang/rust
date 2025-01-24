//@ compile-flags: -Z unstable-options --show-coverage --output-format=json
//@ check-pass

mod bar {
    /// a
    ///
    /// ```
    /// let x = 0;
    /// ```
    pub struct Foo;
}

pub use bar::Foo;
