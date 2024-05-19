// This test ensures that warnings are working as expected for "custom_code_classes_in_docs"
// feature.

#![feature(custom_code_classes_in_docs)]
#![deny(warnings)]
#![feature(no_core)]
#![no_core]

/// ```{class="}
/// main;
/// ```
//~^^^ ERROR unclosed quote string
//~| ERROR unclosed quote string
/// ```"
/// main;
/// ```
pub fn foo() {}
