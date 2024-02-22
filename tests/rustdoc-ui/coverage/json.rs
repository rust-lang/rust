//@ check-pass
//@ compile-flags:-Z unstable-options --output-format json --show-coverage

pub mod foo {
    /// Hello!
    pub struct Foo;
    /// Bar
    pub enum Bar { A }
}

/// X
pub struct X;

/// Bar
///
/// ```
/// let x = 12;
/// ```
pub mod bar {
    /// bar
    pub struct Bar;
    /// X
    pub enum X {
        /// ```
        /// let x = "should be ignored!";
        /// ```
        Y
    }
}

/// yolo
///
/// ```text
/// should not be counted as a code example!
/// ```
pub enum Yolo { X }

impl Yolo {
    /// ```
    /// let x = "should be ignored!";
    /// ```
    pub const Const: u32 = 0;
}

pub struct Xo<T: Clone> {
    /// ```
    /// let x = "should be ignored!";
    /// ```
    x: T,
}

/// ```
/// let x = "should be ignored!";
/// ```
pub static StaticFoo: u32 = 0;

/// ```
/// let x = "should be ignored!";
/// ```
pub const ConstFoo: u32 = 0;

/// ```
/// let x = "should be ignored!";
/// ```
pub type TypeFoo = u32;
