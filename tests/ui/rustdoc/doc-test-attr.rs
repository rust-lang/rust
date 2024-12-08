#![crate_type = "lib"]
#![deny(invalid_doc_attributes)]

#![doc(test)]
//~^ ERROR `#[doc(test(...)]` takes a list of attributes
#![doc(test = "hello")]
//~^ ERROR `#[doc(test(...)]` takes a list of attributes
#![doc(test(a))]
//~^ ERROR unknown `doc(test)` attribute `a`

pub fn foo() {}
