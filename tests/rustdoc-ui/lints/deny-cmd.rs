//@ revisions: deny allow
//@[deny] compile-flags: -Dmissing_docs
//@[allow] compile-flags: -Amissing_docs
//@[allow] check-pass

//! Verify that the `-D` flag, passed to rustdoc, works as expected

pub struct Foo;
//[deny]~^ ERROR missing_docs
