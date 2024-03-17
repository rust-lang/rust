// Check that two-variant fieldless enum with unsigned integer representations don't get represented
// as `i1` except for the bool special case.
//
//@ compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(repr128)]

#[repr(u8)]
pub enum FakeBoolU8 { True, False }

// CHECK: define void @fake_bool_u8(i1 noundef zeroext %_x)
#[no_mangle]
pub fn fake_bool_u8(_x: FakeBoolU8) {}

#[repr(u16)]
pub enum FakeBoolU16 { True, False }

// CHECK: define void @fake_bool_u16(i16 noundef %_x)
#[no_mangle]
pub fn fake_bool_u16(_x: FakeBoolU16) {}

#[repr(u32)]
pub enum FakeBoolU32 { True, False }

// CHECK: define void @fake_bool_u32(i32 noundef %_x)
#[no_mangle]
pub fn fake_bool_u32(_x: FakeBoolU32) {}

#[repr(u64)]
pub enum FakeBoolU64 { True, False }

// CHECK: define void @fake_bool_u64(i64 noundef %_x)
#[no_mangle]
pub fn fake_bool_u64(_x: FakeBoolU64) {}

#[repr(u128)]
pub enum FakeBoolU128 { True, False }

// CHECK: define void @fake_bool_u128(i128 noundef %_x)
#[no_mangle]
pub fn fake_bool_u128(_x: FakeBoolU128) {}

// Check that `zeroext` is present when `extern "C"`.
// CHECK: define void @repr_c_fake_bool_u16(i16 noundef zeroext %_x)
#[no_mangle]
pub extern "C" fn repr_c_fake_bool_u16(_x: FakeBoolU16) {}

pub enum RustFakeBoolU16 { True, False }

// CHECK: define void @repr_rust_fake_bool_u16(i1 noundef zeroext %_x)
#[no_mangle]
pub fn repr_rust_fake_bool_u16(_x: RustFakeBoolU16) {}
