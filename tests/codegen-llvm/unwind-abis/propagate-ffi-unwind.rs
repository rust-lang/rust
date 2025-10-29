//@ compile-flags: -C panic=abort -Cno-prepopulate-passes

// Test that Rustic `#[rustc_propagate_ffi_unwind]` functions are considered unwinding even under
// `-C panic=abort`. We disable optimizations to make sure LLVM doesn't infer attributes.

#![feature(rustc_attrs)]
#![crate_type = "lib"]

// CHECK: @caller() unnamed_addr [[ATTR0:#[0-9]+]]
#[no_mangle]
pub fn caller() {
    // CHECK: call void @{{.*core9panicking19panic_cannot_unwind}}
    may_unwind();
}

// This function would typically be in a different crate.
// CHECK: @may_unwind() unnamed_addr [[ATTR1:#[0-9]+]]
#[no_mangle]
#[rustc_propagate_ffi_unwind]
#[inline(never)]
pub fn may_unwind() {}

// CHECK: attributes [[ATTR0]] = { {{.*}}nounwind{{.*}} }
// CHECK-NOT: attributes [[ATTR1]] = { {{.*}}nounwind{{.*}} }
