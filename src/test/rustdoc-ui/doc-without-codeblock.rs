#![deny(missing_doc_code_examples)] //~ ERROR missing code example in this documentation

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
