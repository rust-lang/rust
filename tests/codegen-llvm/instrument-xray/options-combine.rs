// Checks that `-Z instrument-xray` options can be specified multiple times.
//
//@ needs-xray
//@ compile-flags: -Z instrument-xray=skip-exit -Copt-level=0
//@ compile-flags: -Z instrument-xray=instruction-threshold=123 -Copt-level=0
//@ compile-flags: -Z instrument-xray=instruction-threshold=456 -Copt-level=0

#![crate_type = "lib"]

// CHECK:      attributes #{{.*}} "xray-instruction-threshold"="456" "xray-skip-exit"
// CHECK-NOT:  attributes #{{.*}} "xray-instruction-threshold"="123"
pub fn function() {}
