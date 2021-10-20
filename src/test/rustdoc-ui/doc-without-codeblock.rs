#![deny(rustdoc::missing_doc_code_examples)] //~ ERROR missing code example in this documentation

/// Some docs.
//~^ ERROR missing code example in this documentation
pub struct Foo;

/// And then, the princess died.
//~^ ERROR missing code example in this documentation
pub mod foo {
    /// Or maybe not because she saved herself!
    //~^ ERROR missing code example in this documentation
    pub fn bar() {}
}

// This impl is here to ensure the lint isn't emitted for foreign traits implementations.
impl std::ops::Neg for Foo {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self
    }
}
