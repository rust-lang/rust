// compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(c_ffi_const)]

pub fn bar() { unsafe { foo() } }

extern {
    // CHECK-LABEL: declare void @foo()
    // CHECK-SAME: [[ATTRS:#[0-9]+]]
    // CHECK-DAG: attributes [[ATTRS]] = { {{.*}}readnone{{.*}} }
    #[c_ffi_const] pub fn foo();
}
