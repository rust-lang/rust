//@ compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(ffi_pure)]

pub fn bar() {
    unsafe { foo() }
}

extern "C" {
    // CHECK-LABEL: declare{{.*}}void @foo()
    // CHECK-SAME: [[ATTRS:#[0-9]+]]
    // CHECK-DAG: attributes [[ATTRS]] = { {{.*}}memory(read){{.*}} }
    #[ffi_pure]
    pub fn foo();
}
