// compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(ffi_const)]

// CHECK-LABEL: @bar()
#[no_mangle]
pub fn bar() { unsafe { foo() } }


extern {
    // CHECK-LABEL: @foo() unnamed_addr #1
    // CHECK: attributes #1 = { {{.*}}readnone{{.*}} }
    #[no_mangle]
    #[ffi_const]
    pub fn foo();
}
