// This test ensures that warnings are working as expected for "custom_code_classes_in_docs"
// feature.

#![deny(warnings)]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

/// ```{class="}
/// main;
/// ```
//~^^^ ERROR unclosed quote string
//~| ERROR unclosed quote string
/// ```"
/// main;
/// ```
pub fn foo() {}
