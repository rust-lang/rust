// aux-build:proc_macro.rs
// build-aux-docs

// FIXME: if/when proc-macros start exporting their doc attributes across crates, we can turn on
// cross-crate inlining for them

extern crate some_macros;

// @has proc_macro/index.html
// @has - '//a/@href' '../some_macros/macro.some_proc_macro.html'
// @has - '//a/@href' '../some_macros/attr.some_proc_attr.html'
// @has - '//a/@href' '../some_macros/derive.SomeDerive.html'
// @!has proc_macro/macro.some_proc_macro.html
// @!has proc_macro/attr.some_proc_attr.html
// @!has proc_macro/derive.SomeDerive.html
pub use some_macros::{some_proc_macro, some_proc_attr, SomeDerive};
