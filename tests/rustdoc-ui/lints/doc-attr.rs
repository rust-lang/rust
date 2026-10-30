#![crate_type = "lib"]
#![deny(invalid_doc_attributes)]

#[doc(123)]
//~^ ERROR
#[doc("hello", "bar")]
//~^ ERROR
//~| ERROR
fn bar() {}
