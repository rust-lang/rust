//@ edition: 2024
//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[repr(C, align(8))]
pub struct Aligned8 {
    field: u32,
}

// Forming a reference to an aligned field is permitted,
// even when the enclosing aggregate pointer is misaligned.

// CHECK-LABEL: define{{.*}} @projected_reference_read(
// CHECK: load i32, ptr {{.*}}, align 4

#[unsafe(no_mangle)]
pub fn projected_reference_read(input: &[u64; 2]) -> u32 {
    unsafe {
        let base = input.as_ptr().cast::<u8>().add(4).cast::<Aligned8>();

        let field: &u32 = &(*base).field;
        *field
    }
}
