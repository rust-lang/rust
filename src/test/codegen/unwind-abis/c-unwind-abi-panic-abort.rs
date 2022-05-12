// compile-flags: -C panic=abort

// Test that `nounwind` atributes are also applied to extern `C-unwind` Rust functions
// when the code is compiled with `panic=abort`.

#![crate_type = "lib"]
#![feature(c_unwind)]

// CHECK: @rust_item_that_can_unwind() unnamed_addr #0
#[no_mangle]
pub unsafe extern "C-unwind" fn rust_item_that_can_unwind() {
    // CHECK: call void @_ZN4core9panicking15panic_no_unwind
    may_unwind();
}

extern "C-unwind" {
    // CHECK: @may_unwind() unnamed_addr #1
    fn may_unwind();
}

// Now, make sure that the LLVM attributes for this functions are correct.  First, make
// sure that the first item is correctly marked with the `nounwind` attribute:
//
// CHECK: attributes #0 = { {{.*}}nounwind{{.*}} }
//
// Now, check that foreign item is correctly marked without the `nounwind` attribute.
// CHECK-NOT: attributes #1 = { {{.*}}nounwind{{.*}} }
