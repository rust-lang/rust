//@ compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(ffi_const)]

pub fn bar() {
    unsafe { foo() }
}

extern "C" {
    // CHECK-LABEL: declare{{.*}}void @foo()
    // CHECK-SAME: [[ATTRS:#[0-9]+]]
    // CHECK-DAG: attributes [[ATTRS]] = { {{.*}}memory(none){{.*}} }
    #[unsafe(ffi_const)]
    pub fn foo();
}
