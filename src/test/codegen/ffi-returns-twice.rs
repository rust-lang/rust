// compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(ffi_returns_twice)]

pub fn bar() { unsafe { foo() } }

extern {
    #[ffi_returns_twice] pub fn foo();
}
// CHECK: declare void @foo(){{.*}}#1{{.*}}
// CHECK: attributes #1 = { {{.*}}returns_twice{{.*}} }
