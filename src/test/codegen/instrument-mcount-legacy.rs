// ignore-llvm-version: 13.0.0 - 99.99.99
// revisions: opt noopt
// compile-flags: -Z instrument-mcount
// [opt]compile-flags: -O

#![crate_type = "lib"]

// CHECK: attributes #{{.*}} "frame-pointer"="all" "instrument-function-entry-inlined"="{{.*}}mcount{{.*}}"
pub fn foo() {}
