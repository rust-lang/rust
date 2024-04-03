// Check that small array immediates that fits in target pointer size in argument position have
// `noundef` parameter metadata. Note that the `noundef` parameter metadata is only applied if:
// - `!arg.layout.is_unsized() && size <= Pointer(AddressSpace::DATA).size(cx)`
// - optimizations are turned on.
//
// ignore-tidy-linelength
//@ only-64bit (presence of noundef depends on pointer width)
//@ compile-flags: -C no-prepopulate-passes -O
#![crate_type = "lib"]

// CHECK: define noundef i64 @replace_short_array_u64x1(ptr noalias noundef align 8 dereferenceable(8) %r, i64 noundef %0)
#[no_mangle]
pub fn replace_short_array_u64x1(r: &mut [u64; 1], v: [u64; 1]) -> [u64; 1] {
    std::mem::replace(r, v)
}

// CHECK: define noundef i32 @replace_short_array_u32x1(ptr noalias noundef align 4 dereferenceable(4) %r, i32 noundef %0)
#[no_mangle]
pub fn replace_short_array_u32x1(r: &mut [u32; 1], v: [u32; 1]) -> [u32; 1] {
    std::mem::replace(r, v)
}

// CHECK: define noundef i64 @replace_short_array_u32x2(ptr noalias noundef align 4 dereferenceable(8) %r, i64 noundef %0)
#[no_mangle]
pub fn replace_short_array_u32x2(r: &mut [u32; 2], v: [u32; 2]) -> [u32; 2] {
    std::mem::replace(r, v)
}

// CHECK: define noundef i16 @replace_short_array_u16x1(ptr noalias noundef align 2 dereferenceable(2) %r, i16 noundef %0)
#[no_mangle]
pub fn replace_short_array_u16x1(r: &mut [u16; 1], v: [u16; 1]) -> [u16; 1] {
    std::mem::replace(r, v)
}

// CHECK: define noundef i32 @replace_short_array_u16x2(ptr noalias noundef align 2 dereferenceable(4) %r, i32 noundef %0)
#[no_mangle]
pub fn replace_short_array_u16x2(r: &mut [u16; 2], v: [u16; 2]) -> [u16; 2] {
    std::mem::replace(r, v)
}

// CHECK: define noundef i48 @replace_short_array_u16x3(ptr noalias noundef align 2 dereferenceable(6) %r, i48 noundef %0)
#[no_mangle]
pub fn replace_short_array_u16x3(r: &mut [u16; 3], v: [u16; 3]) -> [u16; 3] {
    std::mem::replace(r, v)
}

// CHECK: define noundef i64 @replace_short_array_u16x4(ptr noalias noundef align 2 dereferenceable(8) %r, i64 noundef %0)
#[no_mangle]
pub fn replace_short_array_u16x4(r: &mut [u16; 4], v: [u16; 4]) -> [u16; 4] {
    std::mem::replace(r, v)
}

// CHECK: define noundef i8 @replace_short_array_u8x1(ptr noalias noundef align 1 dereferenceable(1) %r, i8 noundef %0)
#[no_mangle]
pub fn replace_short_array_u8x1(r: &mut [u8; 1], v: [u8; 1]) -> [u8; 1] {
    std::mem::replace(r, v)
}

// CHECK: define noundef i16 @replace_short_array_u8x2(ptr noalias noundef align 1 dereferenceable(2) %r, i16 noundef %0)
#[no_mangle]
pub fn replace_short_array_u8x2(r: &mut [u8; 2], v: [u8; 2]) -> [u8; 2] {
    std::mem::replace(r, v)
}

// CHECK: define noundef i24 @replace_short_array_u8x3(ptr noalias noundef align 1 dereferenceable(3) %r, i24 noundef %0)
#[no_mangle]
pub fn replace_short_array_u8x3(r: &mut [u8; 3], v: [u8; 3]) -> [u8; 3] {
    std::mem::replace(r, v)
}

// CHECK: define noundef i64 @replace_short_array_u8x8(ptr noalias noundef align 1 dereferenceable(8) %r, i64 noundef %0)
#[no_mangle]
pub fn replace_short_array_u8x8(r: &mut [u8; 8], v: [u8; 8]) -> [u8; 8] {
    std::mem::replace(r, v)
}
