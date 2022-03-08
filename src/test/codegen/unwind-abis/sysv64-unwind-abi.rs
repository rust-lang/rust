// needs-llvm-components: x86
// compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib -Cno-prepopulate-passes
#![no_core]
#![feature(no_core, lang_items, c_unwind)]
#[lang="sized"]
trait Sized { }

// Test that `nounwind` atributes are correctly applied to exported `sysv64` and
// `sysv64-unwind` extern functions. `sysv64-unwind` functions MUST NOT have this attribute. We
// disable optimizations above to prevent LLVM from inferring the attribute.

// CHECK: @rust_item_that_cannot_unwind() unnamed_addr #0 {
#[no_mangle]
pub extern "sysv64" fn rust_item_that_cannot_unwind() {
}

// CHECK: @rust_item_that_can_unwind() unnamed_addr #1 {
#[no_mangle]
pub extern "sysv64-unwind" fn rust_item_that_can_unwind() {
}

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
