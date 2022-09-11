// This tests that the shadowcallstack attribute is
// applied when enabling the shadow-call-stack sanitizer.
//
// needs-sanitizer-shadow-call-stack
// compile-flags: -Zsanitizer=shadow-call-stack

#![crate_type = "lib"]
#![feature(no_sanitize)]

// CHECK: ; Function Attrs:{{.*}}shadowcallstack
// CHECK-NEXT: scs
pub fn scs() {}

// CHECK-NOT: ; Function Attrs:{{.*}}shadowcallstack
// CHECK-NEXT: no_scs
#[no_sanitize(shadow_call_stack)]
pub fn no_scs() {}
