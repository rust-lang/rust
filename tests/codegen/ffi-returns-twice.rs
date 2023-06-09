// compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(ffi_returns_twice)]

pub fn bar() { unsafe { foo() } }

extern "C" {
    // CHECK: declare{{( dso_local)?}} void @foo(){{.*}}[[ATTRS:#[0-9]+]]
    // CHECK: attributes [[ATTRS]] = { {{.*}}returns_twice{{.*}} }
    #[ffi_returns_twice] pub fn foo();
}
