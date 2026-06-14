//
//@ compile-flags: -C instrument-mcount=yes -Copt-level=0

#![crate_type = "lib"]

// CHECK: attributes #{{.*}} "frame-pointer"="all" "instrument-function-entry-inlined"="{{.*}}mcount{{.*}}"
pub fn foo() {}
