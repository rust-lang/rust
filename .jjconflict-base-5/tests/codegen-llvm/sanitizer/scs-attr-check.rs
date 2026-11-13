// This tests that the shadowcallstack attribute is
// applied when enabling the shadow-call-stack sanitizer.
//
//@ needs-sanitizer-shadow-call-stack
//@ compile-flags: -Zsanitizer=shadow-call-stack

#![crate_type = "lib"]
#![feature(sanitize)]

// CHECK: ; sanitizer_scs_attr_check::scs
// CHECK-NEXT: ; Function Attrs:{{.*}}shadowcallstack
pub fn scs() {}

// CHECK: ; sanitizer_scs_attr_check::no_scs
// CHECK-NOT: ; Function Attrs:{{.*}}shadowcallstack
#[sanitize(shadow_call_stack = "off")]
pub fn no_scs() {}
