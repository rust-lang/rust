//@ compile-flags: -C no-prepopulate-passes -Copt-level=0
//
//@ only-x86_64

#![crate_type = "lib"]

extern "C" {
    fn variadic_fn(_: i32, ...);
}

#[repr(C)]
struct Foo(u8);
#[repr(C)]
struct Bar(u64, u64, u64);

// Ensure that emit arguments of the correct type.
pub unsafe fn test_call_variadic() {
    // CHECK: call void (i32, ...) @variadic_fn(i32 0, i8 {{.*}}, ptr {{.*}})
    variadic_fn(0, Foo(0), Bar(0, 0, 0))
}
