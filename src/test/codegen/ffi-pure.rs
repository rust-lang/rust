// compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(ffi_pure)]

pub fn bar() { unsafe { foo() } }

extern {
    #[ffi_pure] pub fn foo();
}
// CHECK: declare void @foo(){{.*}}#1{{.*}}
// CHECK: attributes #1 = { {{.*}}readonly{{.*}} }
