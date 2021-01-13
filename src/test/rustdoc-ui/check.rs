// check-pass
// compile-flags: -Z unstable-options --check

#![warn(missing_docs)]
//~^ WARN
//~^^ WARN
#![warn(rustdoc)]

pub fn foo() {}
//~^ WARN
//~^^ WARN
