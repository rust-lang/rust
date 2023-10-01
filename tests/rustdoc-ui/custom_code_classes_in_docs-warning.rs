// This test ensures that warnings are working as expected for "custom_code_classes_in_docs"
// feature.

#![feature(custom_code_classes_in_docs)]
#![deny(warnings)]
#![feature(no_core)]
#![no_core]

/// ```{. }
/// main;
/// ```
//~^^^ ERROR unexpected ` ` character after `.`
pub fn foo() {}

/// ```{class= a}
/// main;
/// ```
//~^^^ ERROR unexpected ` ` character after `=`
pub fn foo2() {}

/// ```{#id}
/// main;
/// ```
//~^^^ ERROR unexpected character `#`
pub fn foo3() {}

/// ```{{
/// main;
/// ```
//~^^^ ERROR unexpected character `{`
pub fn foo4() {}

/// ```}
/// main;
/// ```
//~^^^ ERROR unexpected character `}`
pub fn foo5() {}

/// ```)
/// main;
/// ```
//~^^^ ERROR unexpected character `)`
pub fn foo6() {}

/// ```{class=}
/// main;
/// ```
//~^^^ ERROR unexpected `}` character after `=`
pub fn foo7() {}

/// ```(
/// main;
/// ```
//~^^^ ERROR unclosed comment: missing `)` at the end
pub fn foo8() {}

/// ```{class=one=two}
/// main;
/// ```
//~^^^ ERROR unexpected `=` character
pub fn foo9() {}

/// ```{.one.two}
/// main;
/// ```
pub fn foo10() {}

/// ```{class=(one}
/// main;
/// ```
//~^^^ ERROR unexpected `(` character after `=`
pub fn foo11() {}

/// ```{class=one.two}
/// main;
/// ```
pub fn foo12() {}

/// ```{(comment)}
/// main;
/// ```
//~^^^ ERROR unexpected character `(`
pub fn foo13() {}
