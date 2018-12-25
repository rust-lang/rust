//~ ERROR Missing code example in this documentation

#![deny(missing_doc_code_examples)]

/// Some docs.
//~^ ERROR Missing code example in this documentation
pub struct Foo;

/// And then, the princess died.
//~^ ERROR Missing code example in this documentation
pub mod foo {
    /// Or maybe not because she saved herself!
    //~^ ERROR Missing code example in this documentation
    pub fn bar() {}
}
