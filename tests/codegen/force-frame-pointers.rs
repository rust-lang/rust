//@ revisions: always nonleaf
//@ [always] compile-flags: -Cforce-frame-pointers=yes
//@ [nonleaf] compile-flags: -Cforce-frame-pointers=non-leaf
//@ compile-flags: -Zunstable-options
//@ compile-flags: -C no-prepopulate-passes -Copt-level=0
//@ [nonleaf] ignore-illumos
//@ [nonleaf] ignore-openbsd
//@ [nonleaf] ignore-x86
//@ [nonleaf] ignore-x86_64-apple-darwin
//@ [nonleaf] ignore-windows-gnu
//@ [nonleaf] ignore-thumb
// result is platform-dependent based on platform's frame pointer settings

#![crate_type = "lib"]

// CHECK-ALWAYS: attributes #{{.*}} "frame-pointer"="all"
// CHECK-NONLEAF: attributes #{{.*}} "frame-pointer"="non-leaf"
pub fn foo() {}
