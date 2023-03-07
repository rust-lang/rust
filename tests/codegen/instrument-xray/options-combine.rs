// Checks that `-Z instrument-xray` options can be specified multiple times.
//
// needs-xray
// compile-flags: -Z instrument-xray=skip-exit
// compile-flags: -Z instrument-xray=instruction-threshold=123
// compile-flags: -Z instrument-xray=instruction-threshold=456

#![crate_type = "lib"]

// CHECK:      attributes #{{.*}} "xray-instruction-threshold"="456" "xray-skip-exit"
// CHECK-NOT:  attributes #{{.*}} "xray-instruction-threshold"="123"
pub fn function() {}
