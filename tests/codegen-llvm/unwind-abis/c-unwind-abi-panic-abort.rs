//@ compile-flags: -C panic=abort
//@ revisions: NONWASM WASM WASMEXN
//@ [NONWASM] ignore-wasm32
//@ [WASM] only-wasm32
//@ [WASMEXN] only-wasm32
//@ [WASMEXN] compile-flags: -Ctarget-feature=+exception-handling

// Test that `nounwind` attributes are also applied to extern `C-unwind` Rust functions
// when the code is compiled with `panic=abort`.

#![crate_type = "lib"]

// CHECK: @rust_item_that_can_unwind() unnamed_addr [[ATTR0:#[0-9]+]]
#[no_mangle]
pub unsafe extern "C-unwind" fn rust_item_that_can_unwind() {
    // Handle both legacy and v0 symbol mangling.
    // NONWASM: call void @{{.*core9panicking19panic_cannot_unwind}}
    // WASMEXN: call void @{{.*core9panicking19panic_cannot_unwind}}
    // WASM-NOT: call void @{{.*core9panicking19panic_cannot_unwind}}
    may_unwind();
}

extern "C-unwind" {
    // CHECK: @may_unwind() unnamed_addr [[ATTR1:#[0-9]+]]
    fn may_unwind();
}

// Now, make sure that the LLVM attributes for this functions are correct.  First, make
// sure that the first item is correctly marked with the `nounwind` attribute:
//
// CHECK: attributes [[ATTR0]] = { {{.*}}nounwind{{.*}} }
//
// Now, check that foreign item is correctly marked without the `nounwind` attribute.
// CHECK-NOT: attributes [[ATTR1]] = { {{.*}}nounwind{{.*}} }
