//@ compile-flags: -Z unstable-options --show-coverage -o -
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
