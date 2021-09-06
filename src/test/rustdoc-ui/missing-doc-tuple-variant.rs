// compile-flags: -Z unstable-options --check

#![deny(missing_docs)]

//! crate level doc

/// Enum doc.
pub enum Foo {
    /// Variant doc.
    Foo(String),
    /// Variant Doc.
    Bar(String, u32),
    //~^ ERROR
    //~^^ ERROR
}
