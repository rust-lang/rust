//@ edition: 2024
//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[repr(C, align(8))]
pub struct Aligned8 {
    field: u32,
}

// A raw field pointer may be aligned to 4 bytes even when
// the enclosing aggregate requires 8-byte alignment.

// CHECK-LABEL: define{{.*}} @raw_mut_field_write(
// CHECK: store i32 42, ptr {{.*}}, align 4

#[unsafe(no_mangle)]
pub fn raw_mut_field_write(input: &mut [u64; 2]) {
    unsafe {
        let ptr = input.as_mut_ptr().cast::<u8>().add(4).cast::<Aligned8>();

        let field = &raw mut (*ptr).field;
        field.write(42);
    }
}
