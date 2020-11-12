// compile-flags: -Z unstable-options --check

#![deny(missing_docs)]
//~^ ERROR
//~^^ ERROR
#![deny(rustdoc)]

pub fn foo() {}
//~^ ERROR
//~^^ ERROR
