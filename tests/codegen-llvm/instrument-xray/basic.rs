// Checks that `-Z instrument-function=xray -Z instrument-xray-opts`
// produces expected instrumentation.
//
//@ needs-xray
//@ compile-flags: -Z instrument-function=xray -Z instrument-xray-opts=always -Copt-level=0

#![crate_type = "lib"]

// CHECK: attributes #{{.*}} "function-instrument"="xray-always"
pub fn function() {}
