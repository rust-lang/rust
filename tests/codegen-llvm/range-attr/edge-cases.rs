#![feature(rustc_attrs)]
#![crate_type = "lib"]
//@ edition: 2024

// Edge-case tests for the conversion from `rustc_abi::WrappingRange` to
// LLVM range attributes.

#[rustc_layout_scalar_valid_range_start(1)]
pub struct LowNiche8(u8);
// CHECK: define void @low_niche_8(i8 noundef range(i8 1, 0) %_1)
#[unsafe(no_mangle)]
pub fn low_niche_8(_: LowNiche8) {}

#[rustc_layout_scalar_valid_range_end(254)]
pub struct HighNiche8(u8);
// CHECK: define void @high_niche_8(i8 noundef range(i8 0, -1) %_1)
#[unsafe(no_mangle)]
pub fn high_niche_8(_: HighNiche8) {}

#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_layout_scalar_valid_range_end(254)]
pub struct Niches8(u8);
// CHECK: define void @niches_8(i8 noundef range(i8 1, -1) %_1)
#[unsafe(no_mangle)]
pub fn niches_8(_: Niches8) {}

#[rustc_layout_scalar_valid_range_start(255)]
#[rustc_layout_scalar_valid_range_end(255)]
pub struct SoloHigh8(u8);
// CHECK: define void @solo_high_8(i8 noundef range(i8 -1, 0) %_1)
#[unsafe(no_mangle)]
pub fn solo_high_8(_: SoloHigh8) {}

#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_layout_scalar_valid_range_end(340282366920938463463374607431768211454)] // (u128::MAX - 1)
pub struct Niches128(u128);
// CHECK: define void @niches_128(i128 noundef range(i128 1, -1) %_1)
#[unsafe(no_mangle)]
pub fn niches_128(_: Niches128) {}
