//@ revisions: x32 x64
//@[x32] only-x86
//@[x64] only-x86_64
//@ compile-flags: -Z hotpatch

#![crate_type = "lib"]

#[no_mangle]
pub fn foo() {}

// CHECK: @foo() unnamed_addr #0
// CHECK: attributes #0 = { {{.*}} "patchable-function"="prologue-short-redirect" {{.*}}}
