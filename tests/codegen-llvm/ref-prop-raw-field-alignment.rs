// Regression test: reference propagation must not strengthen the
// alignment of a load through a raw pointer to an aggregate field.

//@ edition: 2024
//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[repr(C, align(8))]
pub struct Aligned8 {
    field: u32,
}

// CHECK-LABEL: define{{.*}} @raw_field_read(
// CHECK: load i32, ptr {{.*}}, align 4

#[unsafe(no_mangle)]
pub fn raw_field_read(input: &[u64; 2]) -> u32 {
    unsafe {
        let ptr = input.as_ptr().cast::<u8>().add(4).cast::<Aligned8>();

        let field = &raw const (*ptr).field;
        field.read()
    }
}

// A raw-pointer reborrow must not strengthen field alignment.

// CHECK-LABEL: define{{.*}} @raw_reborrow_field_read(
// CHECK: load i32, ptr {{.*}}, align 4

#[unsafe(no_mangle)]
pub fn raw_reborrow_field_read(input: &[u64; 2]) -> u32 {
    unsafe {
        let ptr = input.as_ptr().cast::<u8>().add(4).cast::<Aligned8>();
        let rebound = &raw const *ptr;
        let field = &raw const (*rebound).field;
        // Keep this function distinct from `raw_field_read` so LLVM does
        // not merge the functions and replace this definition with an alias.
        field.read().wrapping_add(1)
    }
}

// A valid reference guarantees the aggregate's alignment.

// CHECK-LABEL: define{{.*}} @aligned_reference_read(
// CHECK: load i32, ptr {{.*}}, align 8

#[unsafe(no_mangle)]
pub fn aligned_reference_read(value: &Aligned8) -> u32 {
    value.field
}
