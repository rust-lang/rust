// run-rustfix
#![feature(integer_atomics)]
#![allow(unused_variables, clippy::blacklisted_name)]
#![deny(clippy::replace_consts)]

use std::sync::atomic::*;
use std::sync::Once;

#[rustfmt::skip]
fn bad() {
    #[allow(deprecated, unused_imports)]
    use std::sync::ONCE_INIT;
    // Once
    { let foo = ONCE_INIT; };
    // Min
    { let foo = std::isize::MIN; };
    { let foo = std::i8::MIN; };
    { let foo = std::i16::MIN; };
    { let foo = std::i32::MIN; };
    { let foo = std::i64::MIN; };
    { let foo = std::i128::MIN; };
    { let foo = std::usize::MIN; };
    { let foo = std::u8::MIN; };
    { let foo = std::u16::MIN; };
    { let foo = std::u32::MIN; };
    { let foo = std::u64::MIN; };
    { let foo = std::u128::MIN; };
    // Max
    { let foo = std::isize::MAX; };
    { let foo = std::i8::MAX; };
    { let foo = std::i16::MAX; };
    { let foo = std::i32::MAX; };
    { let foo = std::i64::MAX; };
    { let foo = std::i128::MAX; };
    { let foo = std::usize::MAX; };
    { let foo = std::u8::MAX; };
    { let foo = std::u16::MAX; };
    { let foo = std::u32::MAX; };
    { let foo = std::u64::MAX; };
    { let foo = std::u128::MAX; };
}

#[rustfmt::skip]
fn good() {
    // Once
    { let foo = Once::new(); };
    // Atomic
    { let foo = AtomicBool::new(false); };
    { let foo = AtomicIsize::new(0); };
    { let foo = AtomicI8::new(0); };
    { let foo = AtomicI16::new(0); };
    { let foo = AtomicI32::new(0); };
    { let foo = AtomicI64::new(0); };
    { let foo = AtomicUsize::new(0); };
    { let foo = AtomicU8::new(0); };
    { let foo = AtomicU16::new(0); };
    { let foo = AtomicU32::new(0); };
    { let foo = AtomicU64::new(0); };
    // Min
    { let foo = isize::min_value(); };
    { let foo = i8::min_value(); };
    { let foo = i16::min_value(); };
    { let foo = i32::min_value(); };
    { let foo = i64::min_value(); };
    { let foo = i128::min_value(); };
    { let foo = usize::min_value(); };
    { let foo = u8::min_value(); };
    { let foo = u16::min_value(); };
    { let foo = u32::min_value(); };
    { let foo = u64::min_value(); };
    { let foo = u128::min_value(); };
    // Max
    { let foo = isize::max_value(); };
    { let foo = i8::max_value(); };
    { let foo = i16::max_value(); };
    { let foo = i32::max_value(); };
    { let foo = i64::max_value(); };
    { let foo = i128::max_value(); };
    { let foo = usize::max_value(); };
    { let foo = u8::max_value(); };
    { let foo = u16::max_value(); };
    { let foo = u32::max_value(); };
    { let foo = u64::max_value(); };
    { let foo = u128::max_value(); };
}

fn main() {
    bad();
    good();
}
