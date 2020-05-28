// check-pass
// compile-flags:-Z unstable-options --output-format json --show-coverage

pub mod foo {
    /// Hello!
    pub struct Foo;
    /// Bar
    pub enum Bar { A }
}

/// X
pub struct X;

/// Bar
pub mod bar {
    /// bar
    pub struct Bar;
    /// X
    pub enum X { Y }
}

/// yolo
pub enum Yolo { X }

pub struct Xo<T: Clone> {
    x: T,
}
