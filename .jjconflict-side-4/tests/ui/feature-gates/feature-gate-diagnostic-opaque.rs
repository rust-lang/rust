#![crate_type = "lib"]
#![feature(decl_macro)]
#![deny(unknown_diagnostic_attributes)]

#[diagnostic::opaque]
//~^ ERROR unknown diagnostic attribute
macro_rules! foo {
    () => {}
}

#[diagnostic::opaque]
//~^ ERROR unknown diagnostic attribute
macro bar() {}
