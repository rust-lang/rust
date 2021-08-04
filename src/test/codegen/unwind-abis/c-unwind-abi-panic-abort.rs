// compile-flags: -C panic=abort

// Test that `nounwind` atributes are not applied to `C-unwind` extern functions
// even when the code is compiled with `panic=abort`.

#![crate_type = "lib"]
#![feature(c_unwind)]

extern "C-unwind" {
    fn may_unwind();
}

// CHECK: @rust_item_that_can_unwind() unnamed_addr #0
#[no_mangle]
pub unsafe extern "C-unwind" fn rust_item_that_can_unwind() {
    may_unwind();
}

// Now, make sure that the LLVM attributes for this functions are correct.  First, make
// sure that the first item is correctly marked with the `nounwind` attribute:
//
// CHECK-NOT: attributes #0 = { {{.*}}nounwind{{.*}} }
