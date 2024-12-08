//@ needs-llvm-components: x86
//@ compile-flags: --target=i686-pc-windows-msvc --crate-type=rlib -Cno-prepopulate-passes
#![no_core]
#![feature(no_core, lang_items)]
#[lang = "sized"]
trait Sized {}

// Test that `nounwind` attributes are correctly applied to exported `thiscall` and
// `thiscall-unwind` extern functions. `thiscall-unwind` functions MUST NOT have this attribute. We
// disable optimizations above to prevent LLVM from inferring the attribute.

// CHECK: @rust_item_that_cannot_unwind() unnamed_addr #0 {
#[no_mangle]
pub extern "thiscall" fn rust_item_that_cannot_unwind() {}

// CHECK: @rust_item_that_can_unwind() unnamed_addr #1 {
#[no_mangle]
pub extern "thiscall-unwind" fn rust_item_that_can_unwind() {}

// Now, make some assertions that the LLVM attributes for these functions are correct.  First, make
// sure that the first item is correctly marked with the `nounwind` attribute:
//
// CHECK: attributes #0 = { {{.*}}nounwind{{.*}} }
//
// Next, let's assert that the second item, which CAN unwind, does not have this attribute.
//
// CHECK: attributes #1 = {
// CHECK-NOT: nounwind
// CHECK: }
