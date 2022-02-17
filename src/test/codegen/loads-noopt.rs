// compile-flags: -C opt-level=0 -C no-prepopulate-passes

// This test checks that load instructions in opt-level=0 builds,
// while lacking metadata used for optimization, still get `align` attributes.

#![crate_type = "lib"]

pub struct Bytes {
    a: u8,
    b: u8,
    c: u8,
    d: u8,
}

#[derive(Copy, Clone)]
#[repr(align(4))]
pub struct Align4(i16);

// CHECK-LABEL: @load_bool
#[no_mangle]
pub fn load_bool(x: &bool) -> bool {
// CHECK: load i8, i8* %x, align 1
    *x
}

// CHECK-LABEL: small_array_alignment
#[no_mangle]
pub fn small_array_alignment(x: [i8; 4]) -> [i8; 4] {
// CHECK: load i32, i32* %{{.*}}, align 1
    x
}

// CHECK-LABEL: small_struct_alignment
#[no_mangle]
pub fn small_struct_alignment(x: Bytes) -> Bytes {
// CHECK: load i32, i32* %{{.*}}, align 1
    x
}

// CHECK-LABEL: @load_higher_alignment
#[no_mangle]
pub fn load_higher_alignment(x: &Align4) -> Align4 {
// CHECK: load i32, i32* %{{.*}}, align 4
    *x
}
