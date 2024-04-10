// Check that small array immediates that fits in target pointer size in argument position have
// `noundef` parameter metadata. Note that the `noundef` parameter metadata is only applied if:
// - `!arg.layout.is_unsized() && size <= Pointer(AddressSpace::DATA).size(cx)`
// - optimizations are turned on.
//
//@ only-64bit (presence of noundef depends on pointer width)
//@ compile-flags: -C no-prepopulate-passes -O
#![crate_type = "lib"]

// CHECK: define noundef i64 @short_array_u64x1(i64 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u64x1(v: [u64; 1]) -> [u64; 1] {
    v
}

// CHECK: define noundef i32 @short_array_u32x1(i32 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u32x1(v: [u32; 1]) -> [u32; 1] {
    v
}

// CHECK: define noundef i64 @short_array_u32x2(i64 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u32x2(v: [u32; 2]) -> [u32; 2] {
    v
}

// CHECK: define noundef i16 @short_array_u16x1(i16 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u16x1(v: [u16; 1]) -> [u16; 1] {
    v
}

// CHECK: define noundef i32 @short_array_u16x2(i32 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u16x2(v: [u16; 2]) -> [u16; 2] {
    v
}

// CHECK: define noundef i48 @short_array_u16x3(i48 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u16x3(v: [u16; 3]) -> [u16; 3] {
    v
}

// CHECK: define noundef i64 @short_array_u16x4(i64 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u16x4(v: [u16; 4]) -> [u16; 4] {
    v
}

// CHECK: define noundef i8 @short_array_u8x1(i8 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u8x1(v: [u8; 1]) -> [u8; 1] {
    v
}

// CHECK: define noundef i16 @short_array_u8x2(i16 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u8x2(v: [u8; 2]) -> [u8; 2] {
    v
}

// CHECK: define noundef i24 @short_array_u8x3(i24 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u8x3(v: [u8; 3]) -> [u8; 3] {
    v
}

// CHECK: define noundef i64 @short_array_u8x8(i64 noundef %{{.*}})
#[no_mangle]
pub fn short_array_u8x8(v: [u8; 8]) -> [u8; 8] {
    v
}

#[repr(transparent)]
pub struct Foo([u8; 4]);

// CHECK: define noundef i32 @repr_transparent_struct_short_array(i32 noundef %{{.*}})
#[no_mangle]
pub fn repr_transparent_struct_short_array(v: Foo) -> Foo {
    v
}

#[repr(transparent)]
pub enum Bar {
    Default([u8; 4]),
}

// CHECK: define noundef i32 @repr_transparent_enum_short_array(i32 noundef %{{.*}})
#[no_mangle]
pub fn repr_transparent_enum_short_array(v: Bar) -> Bar {
    v
}

#[repr(transparent)]
pub struct Owo([u8; 4]);

#[repr(transparent)]
pub struct Uwu(Owo);

#[repr(transparent)]
pub struct Oowoo(Uwu);

// CHECK: define noundef i32 @repr_transparent_nested_struct_short_array(i32 noundef %{{.*}})
#[no_mangle]
pub fn repr_transparent_nested_struct_short_array(v: Oowoo) -> Oowoo {
    v
}

// # Negative examples

// This inner struct is *not* `#[repr(transparent)]`, so we must not emit `noundef` for the outer
// struct.
pub struct NotTransparent([u8; 4]);

#[repr(transparent)]
pub struct Transparent(NotTransparent);

// CHECK-LABEL: not_all_transparent_nested_struct_short_array
// CHECK-NOT: noundef
#[no_mangle]
pub fn not_all_transparent_nested_struct_short_array(v: Transparent) -> Transparent {
    v
}
