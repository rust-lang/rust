// This test ensures that warnings are working as expected for "custom_code_classes_in_docs"
// feature.

#![deny(warnings)]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "size_of_val"]
pub trait SizeOfVal: PointeeSized {}

#[lang = "sized"]
pub trait Sized: SizeOfVal {}

/// ```{class="}
/// main;
/// ```
//~^^^ ERROR unclosed quote string
//~| ERROR unclosed quote string
/// ```"
/// main;
/// ```
pub fn foo() {}
