// compile-flags: -C panic=abort -C opt-level=0

// Test that `nounwind` atributes are applied to `C-unwind` extern functions when the
// code is compiled with `panic=abort`.  We disable optimizations above to prevent LLVM from
// inferring the attribute.

#![crate_type = "lib"]
#![feature(c_unwind)]

// CHECK: @rust_item_that_can_unwind() unnamed_addr #0 {
#[no_mangle]
pub extern "C-unwind" fn rust_item_that_can_unwind() {
}

// Now, make sure that the LLVM attributes for this functions are correct.  First, make
// sure that the first item is correctly marked with the `nounwind` attribute:
//
// CHECK: attributes #0 = { {{.*}}nounwind{{.*}} }
