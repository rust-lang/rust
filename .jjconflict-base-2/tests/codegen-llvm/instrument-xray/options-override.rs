// Checks that the last `-Z instrument-xray` option wins.
//
//@ needs-xray
//@ compile-flags: -Z instrument-xray=always -Copt-level=0
//@ compile-flags: -Z instrument-xray=never -Copt-level=0

#![crate_type = "lib"]

// CHECK:      attributes #{{.*}} "function-instrument"="xray-never"
// CHECK-NOT:  attributes #{{.*}} "function-instrument"="xray-always"
pub fn function() {}
