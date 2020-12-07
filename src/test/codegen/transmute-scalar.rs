// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

// CHECK: define i32 @f32_to_bits(float %x)
// CHECK: %2 = bitcast float %x to i32
// CHECK-NEXT: store i32 %2, i32* %0
// CHECK-NEXT: %3 = load i32, i32* %0
// CHECK: ret i32 %3
#[no_mangle]
pub fn f32_to_bits(x: f32) -> u32 {
    unsafe { std::mem::transmute(x) }
}

// CHECK: define i8 @bool_to_byte(i1 zeroext %b)
// CHECK: %1 = zext i1 %b to i8
// CHECK-NEXT: store i8 %1, i8* %0
// CHECK-NEXT: %2 = load i8, i8* %0
// CHECK: ret i8 %2
#[no_mangle]
pub fn bool_to_byte(b: bool) -> u8 {
    unsafe { std::mem::transmute(b) }
}

// CHECK: define zeroext i1 @byte_to_bool(i8 %byte)
// CHECK: %1 = trunc i8 %byte to i1
// CHECK-NEXT: %2 = zext i1 %1 to i8
// CHECK-NEXT: store i8 %2, i8* %0
// CHECK-NEXT: %3 = load i8, i8* %0
// CHECK-NEXT: %4 = trunc i8 %3 to i1
// CHECK: ret i1 %4
#[no_mangle]
pub unsafe fn byte_to_bool(byte: u8) -> bool {
    std::mem::transmute(byte)
}
