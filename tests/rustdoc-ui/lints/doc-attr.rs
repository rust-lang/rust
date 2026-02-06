#![crate_type = "lib"]
#![deny(invalid_doc_attributes)]

#[doc(123)]
//~^ ERROR malformed `doc` attribute
//~| WARN
#[doc("hello", "bar")]
//~^ ERROR malformed `doc` attribute
//~| ERROR malformed `doc` attribute
//~| WARN
//~| WARN
fn bar() {}
