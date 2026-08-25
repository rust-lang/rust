//
//@ compile-flags: -Z instrument-mcount -Copt-level=0

#![crate_type = "lib"]

// CHECK: attributes #{{.*}} "frame-pointer"="all" "instrument-function-entry-inlined"="{{.*}}mcount{{.*}}"
pub fn foo() {}
