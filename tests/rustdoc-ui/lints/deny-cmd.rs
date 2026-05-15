//@ revisions: deny allow
//@[deny] compile-flags: -Dmissing_docs
//@[allow] compile-flags: -Amissing_docs
//@[allow] check-pass
//! docs for crate

pub struct Foo;
//[deny]~^ ERROR missing_docs
