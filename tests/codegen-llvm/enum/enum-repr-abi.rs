// Ensure that enums with a `#[repr($int)]` attribute use the same LLVM type as $int.
// An enum with 2 variants and a `#[repr(u8)]` attribute were previously lowering to
// `i1 zeroext` which was dubious.
// Issue: https://github.com/rust-lang/rust/issues/165026

//@ compile-flags: -Cno-prepopulate-passes

#![crate_type = "lib"]

pub enum NoRepr {
    A,
    B,
}

#[repr(u8)]
pub enum ReprU8 {
    A,
    B,
}

#[repr(i8)]
pub enum ReprI8 {
    A,
    B,
}

#[repr(u16)]
pub enum ReprU16 {
    A,
    B,
}

#[repr(i32)]
pub enum ReprI32 {
    A,
    B,
}

#[repr(i64)]
pub enum ReprI64 {
    A,
    B,
}

// CHECK: noundef{{( zeroext)?}} i1 @enum_no_repr(i1 noundef{{( zeroext)?}} %value)
#[unsafe(no_mangle)]
pub extern "C" fn enum_no_repr(value: NoRepr) -> NoRepr {
    // CHECK: start:
    // CHECK-NEXT: ret i1 %value
    value
}

// CHECK: noundef{{( zeroext)?}} range(i8 0, 2) i8 @enum_u8(i8 noundef{{( zeroext)?}} range(i8 0, 2) %value)
#[unsafe(no_mangle)]
pub extern "C" fn enum_u8(value: ReprU8) -> ReprU8 {
    // CHECK: start:
    // CHECK-NEXT: ret i8 %value
    value
}

// CHECK: noundef{{( signext)?}} range(i8 0, 2) i8 @enum_i8(i8 noundef{{( signext)?}} range(i8 0, 2) %value)
#[unsafe(no_mangle)]
pub extern "C" fn enum_i8(value: ReprI8) -> ReprI8 {
    // CHECK: start:
    // CHECK-NEXT: ret i8 %value
    value
}

// CHECK: noundef{{( zeroext)?}} range(i16 0, 2) i16 @enum_u16(i16 noundef{{( zeroext)?}} range(i16 0, 2) %value)
#[unsafe(no_mangle)]
pub extern "C" fn enum_u16(value: ReprU16) -> ReprU16 {
    // CHECK: start:
    // CHECK-NEXT: ret i16 %value
    value
}

// CHECK: noundef range(i32 0, 2) i32 @enum_i32(i32 noundef range(i32 0, 2) %value)
#[unsafe(no_mangle)]
pub extern "C" fn enum_i32(value: ReprI32) -> ReprI32 {
    // CHECK: start:
    // CHECK-NEXT: ret i32 %value
    value
}

// CHECK: noundef range(i64 0, 2) i64 @enum_i64(i64 noundef range(i64 0, 2) %value)
#[unsafe(no_mangle)]
pub extern "C" fn enum_i64(value: ReprI64) -> ReprI64 {
    // CHECK: start:
    // CHECK-NEXT: ret i64 %value
    value
}
