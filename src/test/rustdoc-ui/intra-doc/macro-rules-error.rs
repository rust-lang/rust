// `macro_rules` scopes are respected during doc link resolution.

// compile-flags: --document-private-items

#![deny(rustdoc::broken_intra_doc_links)]

mod no_escape {
    macro_rules! before_but_limited_to_module {
        () => {};
    }
}

/// [before_but_limited_to_module] FIXME: This error should be reported
// ERROR unresolved link to `before_but_limited_to_module`
/// [after] FIXME: This error should be reported
// ERROR unresolved link to `after`
/// [str] FIXME: This error shouldn not be reported
//~^ ERROR `str` is both a builtin type and a macro
fn check() {}

macro_rules! after {
    () => {};
}

macro_rules! str {
    () => {};
}
