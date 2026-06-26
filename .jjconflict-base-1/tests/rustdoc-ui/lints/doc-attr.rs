#![crate_type = "lib"]
#![deny(invalid_doc_attributes)]

#[doc(123)]
//~^ ERROR
//~| WARN
#[doc("hello", "bar")]
//~^ ERROR
//~| ERROR
//~| WARN
//~| WARN
fn bar() {}
