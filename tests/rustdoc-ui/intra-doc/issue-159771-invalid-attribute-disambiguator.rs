#![feature(macro_attr)]
#![deny(rustdoc::broken_intra_doc_links)]
//~^ NOTE lint level is defined

#[macro_export]
macro_rules! foo {
    attr() () => {};
}

pub fn foo() {}

/// Ambiguous link [foo].
//~^ ERROR `foo` is both a
//~| NOTE ambiguous link
//~| HELP to link to the function
//~| HELP to link to the attribute macro, prefix with `attribute@`

/// Link to my attribute [attribute@foo]
pub fn f() {}
