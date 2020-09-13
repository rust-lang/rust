#![deny(missing_doc_code_examples)]

/// Some docs.
//~^ ERROR missing code example in this documentation
pub struct Foo;

/// And then, the princess died.
pub mod foo {
    /// Or maybe not because she saved herself!
    //~^ ERROR missing code example in this documentation
    pub fn bar() {}
}
