//@ edition: 2024
//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[repr(C)]
pub struct Inner {
    value: u32,
}

#[repr(C, align(8))]
pub struct Outer {
    inner: Inner,
}

// CHECK-LABEL: define{{.*}} @raw_nested_field_read(
// CHECK: load i32, ptr {{.*}}, align 4

#[unsafe(no_mangle)]
pub fn raw_nested_field_read(input: &[u64; 2]) -> u32 {
    unsafe {
        let base = input.as_ptr().cast::<u8>().add(4).cast::<Outer>();

        let field = &raw const (*base).inner.value;
        let copied_field = field;

        copied_field.read()
    }
}
