// This test ensures that warnings are working as expected for "custom_code_classes_in_docs"
// feature.

#![feature(custom_code_classes_in_docs)]
#![deny(warnings)]
#![feature(no_core)]
#![no_core]

/// ```{. class= whatever=hehe #id} } {{
/// main;
/// ```
//~^^^ ERROR missing class name after `.`
//~| ERROR missing class name after `class=`
//~| ERROR unsupported attribute `whatever=hehe`
//~| ERROR unsupported attribute `#id`
//~| ERROR unexpected `}` outside attribute block (`{}`)
//~| ERROR unclosed attribute block (`{}`): missing `}` at the end
//~| ERROR unexpected `{` inside attribute block (`{}`)
pub fn foo() {}
