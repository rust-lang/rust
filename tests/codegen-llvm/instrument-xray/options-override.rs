// Checks that the last `-Z instrument-xray-opts` option wins.
//
//@ needs-xray
//@ compile-flags: -Z instrument-function=xray -Z instrument-xray-opts=always -Copt-level=0
//@ compile-flags: -Z instrument-function=xray -Z instrument-xray-opts=never -Copt-level=0

#![crate_type = "lib"]

// CHECK:      attributes #{{.*}} "function-instrument"="xray-never"
// CHECK-NOT:  attributes #{{.*}} "function-instrument"="xray-always"
pub fn function() {}
