// compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(ffi_const)]

pub fn bar() { unsafe { foo() } }

extern {
    #[ffi_const] pub fn foo();
}
// CHECK: declare void @foo(){{.*}}#1{{.*}}
// CHECK: attributes #1 = { {{.*}}readnone{{.*}} }
