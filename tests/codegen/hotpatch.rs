// check if functions get the attribute, so that LLVM ensures they are hotpatchable
// the attribute is only implemented for x86, aarch64 does not require it

//@ revisions: x32 x64
//@[x32] only-x86
//@[x64] only-x86_64
//@ compile-flags: -Z hotpatch

#![crate_type = "lib"]

#[no_mangle]
pub fn foo() {}

// CHECK-LABEL: @foo() unnamed_addr #0
// CHECK: attributes #0 = { {{.*}} "patchable-function"="prologue-short-redirect" {{.*}}}
