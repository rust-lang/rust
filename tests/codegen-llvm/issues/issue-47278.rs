// -C no-prepopulate-passes
#![crate_type = "staticlib"]

#[repr(C)]
pub struct Foo(u64);

// CHECK: define {{.*}} @foo(
#[no_mangle]
pub extern "C" fn foo(_: Foo) -> Foo {
    loop {}
}
