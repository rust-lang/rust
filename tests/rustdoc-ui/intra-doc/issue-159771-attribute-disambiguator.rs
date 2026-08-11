//@ edition:2024
#![feature(macro_attr)]
#![deny(rustdoc::broken_intra_doc_links)]
//~^ NOTE lint level is defined

// Regression test for https://github.com/rust-lang/rust/issues/159771:
// rustdoc suggested the `attribute@` disambiguator for attribute macros, but did
// not accept it when parsing links.

mod example {}

#[macro_export]
macro_rules! example {
    attr() () => {};
}

/// Wrong bang disambiguator: [example!]
//~^ ERROR incompatible link kind for `example`
//~| NOTE this link resolved to an attribute macro
//~| HELP prefix with `attribute@`

/// Correct attribute disambiguator: [attribute@example]

/// Macro namespace also works: [macro@example]
pub fn f() {}
