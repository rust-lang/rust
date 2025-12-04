#![crate_type = "lib"]

#[doc(123)]
//~^ ERROR malformed `doc` attribute
#[doc("hello", "bar")]
//~^ ERROR malformed `doc` attribute
//~| ERROR malformed `doc` attribute
fn bar() {}
