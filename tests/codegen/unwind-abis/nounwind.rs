//@ compile-flags: -C opt-level=0 -Cpanic=abort
//@ needs-unwind

#![crate_type = "lib"]

// We disable optimizations to prevent LLVM from inferring the attribute.

// CHECK: Function Attrs:{{.*}}nounwind
// CHECK-NEXT: @foo
#[no_mangle]
pub extern "C" fn foo() {}

// CHECK: Function Attrs:{{.*}}nounwind
// CHECK-NEXT: @bar
#[no_mangle]
pub fn bar() {}
