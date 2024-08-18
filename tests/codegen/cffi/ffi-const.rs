//@ compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(ffi_const)]

#[no_mangle]
pub fn bar() {
    // CHECK-LABEL: @bar
    // CHECK: @foo
    // CHECK-SAME: [[ATTRS:#[0-9]+]]
    unsafe { foo() }
}

extern "C" {
    #[ffi_const]
    pub fn foo();
}

// The attribute changed from `readnone` to `memory(none)` with LLVM 16.0.
// CHECK: attributes [[ATTRS]] = { {{.*}}{{readnone|memory\(none\)}}{{.*}} }
