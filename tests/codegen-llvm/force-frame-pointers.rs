//@ revisions: Always NonLeaf
//@ [Always] compile-flags: -Cforce-frame-pointers=yes
//@ [NonLeaf] compile-flags: -Cforce-frame-pointers=non-leaf
//@ compile-flags: -Zunstable-options
//@ compile-flags: -C no-prepopulate-passes -Copt-level=0
//@ [NonLeaf] ignore-illumos
//@ [NonLeaf] ignore-openbsd
//@ [NonLeaf] ignore-x86
//@ [NonLeaf] ignore-x86_64-apple-darwin
//@ [NonLeaf] ignore-windows-gnu
//@ [NonLeaf] ignore-thumb
// result is platform-dependent based on platform's frame pointer settings

#![crate_type = "lib"]

// Always: attributes #{{.*}} "frame-pointer"="all"
// NonLeaf: attributes #{{.*}} "frame-pointer"="non-leaf"
pub fn foo() {}
