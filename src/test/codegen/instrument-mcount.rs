// min-llvm-version 8.0
// ignore-tidy-linelength
// compile-flags: -Z instrument-mcount

#![crate_type = "lib"]

// CHECK: attributes #{{.*}} "frame-pointer"="all" "instrument-function-entry-inlined"="{{.*}}mcount{{.*}}"
pub fn foo() {}
