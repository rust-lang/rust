//@ revisions: Always NonLeaf
//@ [Always] compile-flags: -Cforce-frame-pointers=yes
//@ [NonLeaf] compile-flags: -Cforce-frame-pointers=non-leaf
//@ compile-flags: -Zunstable-options
//@ compile-flags: -C no-prepopulate-passes -Copt-level=0

#![crate_type = "lib"]

// Always: attributes #{{.*}} "frame-pointer"="all"
// NonLeaf: attributes #{{.*}} "frame-pointer"="non-leaf"
pub fn foo() {}
