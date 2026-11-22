#![crate_type = "lib"]

#[doc(123)]
//~^ ERROR
#[doc("hello", "bar")]
//~^ ERROR
//~| ERROR
fn bar() {}
