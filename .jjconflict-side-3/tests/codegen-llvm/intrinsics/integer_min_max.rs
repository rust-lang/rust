//@ compile-flags: -C opt-level=3 -C no-prepopulate-passes

#![crate_type = "lib"]

#[unsafe(no_mangle)]
pub fn i16_min(a: i16, b: i16) -> i16 {
    // CHECK-LABEL: i16_min
    // CHECK: [[M:%.+]] = call i16 @llvm.smin.i16(i16 %a, i16 %b)
    // CHECK-NEXT: ret i16 [[M]]
    std::cmp::min(a, b)
}

#[unsafe(no_mangle)]
pub fn i32_max(a: i32, b: i32) -> i32 {
    // CHECK-LABEL: i32_max
    // CHECK: [[M:%.+]] = call i32 @llvm.smax.i32(i32 %a, i32 %b)
    // CHECK-NEXT: ret i32 [[M]]
    std::cmp::max(a, b)
}

#[unsafe(no_mangle)]
pub fn u8_min(a: u8, b: u8) -> u8 {
    // CHECK-LABEL: u8_min
    // CHECK: [[M:%.+]] = call i8 @llvm.umin.i8(i8 %a, i8 %b)
    // CHECK-NEXT: ret i8 [[M]]
    std::cmp::min(a, b)
}

#[unsafe(no_mangle)]
pub fn u16_max(a: u16, b: u16) -> u16 {
    // CHECK-LABEL: u16_max
    // CHECK: [[M:%.+]] = call i16 @llvm.umax.i16(i16 %a, i16 %b)
    // CHECK-NEXT: ret i16 [[M]]
    std::cmp::max(a, b)
}

#[unsafe(no_mangle)]
pub fn char_min(a: char, b: char) -> char {
    // CHECK-LABEL: char_min
    // CHECK: [[M:%.+]] = call i32 @llvm.umin.i32(i32 %a, i32 %b)
    // CHECK: ret i32 [[M]]
    std::cmp::min(a, b)
}

#[unsafe(no_mangle)]
pub fn char_max(a: char, b: char) -> char {
    // CHECK-LABEL: char_max
    // CHECK: [[M:%.+]] = call i32 @llvm.umax.i32(i32 %a, i32 %b)
    // CHECK: ret i32 [[M]]
    std::cmp::max(a, b)
}
