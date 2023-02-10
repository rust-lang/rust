// Checks that the last `-Z instrument-xray` option wins.
//
// needs-xray
// compile-flags: -Z instrument-xray=always
// compile-flags: -Z instrument-xray=never

#![crate_type = "lib"]

// CHECK:      attributes #{{.*}} "function-instrument"="xray-never"
// CHECK-NOT:  attributes #{{.*}} "function-instrument"="xray-always"
pub fn function() {}
