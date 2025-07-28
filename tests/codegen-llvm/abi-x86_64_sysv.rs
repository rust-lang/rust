//@ only-x86_64

//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

pub struct S24 {
    a: i8,
    b: i8,
    c: i8,
}

pub struct S48 {
    a: i16,
    b: i16,
    c: i8,
}

// CHECK: i24 @struct_24_bits(i24
#[no_mangle]
pub extern "sysv64" fn struct_24_bits(a: S24) -> S24 {
    a
}

// CHECK: i48 @struct_48_bits(i48
#[no_mangle]
pub extern "sysv64" fn struct_48_bits(a: S48) -> S48 {
    a
}
