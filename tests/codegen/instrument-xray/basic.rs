// Checks that `-Z instrument-xray` produces expected instrumentation.
//
// needs-xray
// compile-flags: -Z instrument-xray=always

#![crate_type = "lib"]

// CHECK: attributes #{{.*}} "function-instrument"="xray-always"
pub fn function() {}
