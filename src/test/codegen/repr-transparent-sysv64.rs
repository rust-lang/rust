// only-x86_64

// compile-flags: -C no-prepopulate-passes

#![crate_type="lib"]

#[repr(C)]
pub struct Rgb8 { r: u8, g: u8, b: u8 }

#[repr(transparent)]
pub struct Rgb8Wrap(Rgb8);

// CHECK: i24 @test_Rgb8Wrap(i24)
#[no_mangle]
pub extern "sysv64" fn test_Rgb8Wrap(_: Rgb8Wrap) -> Rgb8Wrap { loop {} }

#[repr(C)]
pub union FloatBits {
    float: f32,
    bits: u32,
}

#[repr(transparent)]
pub struct SmallUnion(FloatBits);

// CHECK: i32 @test_SmallUnion(i32)
#[no_mangle]
pub extern "sysv64" fn test_SmallUnion(_: SmallUnion) -> SmallUnion { loop {} }
