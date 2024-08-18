//@ compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(ffi_pure)]

#[no_mangle]
pub fn bar() {
    // CHECK-LABEL: @bar
    // CHECK: @foo
    // CHECK-SAME: [[ATTRS:#[0-9]+]]
    unsafe { foo() }
}

extern "C" {
    #[ffi_pure]
    pub fn foo();
}

// The attribute changed from `readonly` to `memory(read)` with LLVM 16.0.
// CHECK: attributes [[ATTRS]] = { {{.*}}{{readonly|memory\(read\)}}{{.*}} }
