// ignore-tidy-linelength
// compile-flags: -Z instrument-mcount

#![crate_type = "lib"]

// CHECK: attributes #{{.*}} "instrument-function-entry-inlined"="{{_*}}mcount" "no-frame-pointer-elim"="true"
pub fn foo() {}
