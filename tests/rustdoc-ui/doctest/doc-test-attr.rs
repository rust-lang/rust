#![crate_type = "lib"]
#![deny(invalid_doc_attributes)]

#![doc(test)]
//~^ ERROR `#[doc(test(...)]` takes a list of attributes
//~^^ WARN this was previously accepted by the compiler
#![doc(test = "hello")]
//~^ ERROR `#[doc(test(...)]` takes a list of attributes
//~^^ WARN this was previously accepted by the compiler
#![doc(test(a))]
//~^ ERROR unknown `doc(test)` attribute `a`
//~^^ WARN this was previously accepted by the compiler

pub fn foo() {}
