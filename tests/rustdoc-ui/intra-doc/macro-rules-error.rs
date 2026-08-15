// `macro_rules` scopes are respected during doc link resolution.

//@ compile-flags: --document-private-items

#![deny(rustdoc::broken_intra_doc_links)]

mod no_escape {
    macro_rules! before_but_limited_to_module {
        () => {};
    }
}

/// [before_but_limited_to_module]
//~^ ERROR unresolved link to `before_but_limited_to_module`
/// [after]
//~^ ERROR unresolved link to `after`
/// [str]
fn check() {}

macro_rules! after {
    () => {};
}

macro_rules! str {
    () => {};
}
