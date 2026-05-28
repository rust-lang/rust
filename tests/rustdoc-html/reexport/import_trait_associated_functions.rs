// This test ensures that reexports of associated items links to the associated items.
// Regression test for <https://github.com/rust-lang/rust/issues/148008>.

#![feature(import_trait_associated_functions)]

#![crate_name = "foo"]

//@ has 'foo/index.html'

pub trait Test {
    fn method();
    const CONST: u8;
    type Type;
}

//@ has - '//*[@id="reexport.method"]//a[@href="trait.Test.html#tymethod.method"]' 'method'
//@ has - '//*[@id="reexport.CONST"]//a[@href="trait.Test.html#associatedconstant.CONST"]' 'CONST'
//@ has - '//*[@id="reexport.Type"]//a[@href="trait.Test.html#associatedtype.Type"]' 'Type'
pub use self::Test::{method, CONST, Type};
