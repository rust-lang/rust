// compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(ffi_returns_twice)]

extern {
    // CHECK-LABEL: @foo()
    // CHECK: attributes #1 = { {{.*}}returns_twice{{.*}} }
    #[no_mangle]
    #[ffi_returns_twice]
    pub fn foo();
}

pub fn bar() {
    unsafe { foo() }
}
