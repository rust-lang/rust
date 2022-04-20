// min-llvm-version: 13.0.0
// revisions: opt noopt
// compile-flags: -Z instrument-mcount
// [opt]compile-flags: -O

#![crate_type = "lib"]

// CHECK: call void @{{.*}}mcount
pub fn foo() {}
