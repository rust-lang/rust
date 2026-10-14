// Compiler:
//
// Run-time:
//   status: 0

// Unsigned atomic max/min must compare as unsigned: a backend that compares them as signed keeps
// 0 as the max of 0 and u*::MAX.

use std::hint::black_box;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::{
    AtomicI8, AtomicI16, AtomicI32, AtomicU8, AtomicU16, AtomicU32, AtomicUsize,
};
#[cfg(target_has_atomic = "64")]
use std::sync::atomic::{AtomicI64, AtomicU64};

macro_rules! check_unsigned {
    ($atomic:ident, $int:ident) => {{
        let value = $atomic::new(0);
        assert_eq!(value.fetch_max(black_box($int::MAX), SeqCst), 0);
        assert_eq!(value.load(SeqCst), $int::MAX);
        assert_eq!(value.fetch_max(black_box(1), SeqCst), $int::MAX);
        assert_eq!(value.load(SeqCst), $int::MAX);

        assert_eq!(value.fetch_min(black_box(0), SeqCst), $int::MAX);
        assert_eq!(value.load(SeqCst), 0);
        assert_eq!(value.fetch_min(black_box($int::MAX), SeqCst), 0);
        assert_eq!(value.load(SeqCst), 0);
    }};
}

macro_rules! check_signed {
    ($atomic:ident, $int:ident) => {{
        let value = $atomic::new(0);
        assert_eq!(value.fetch_max(black_box(-1), SeqCst), 0);
        assert_eq!(value.load(SeqCst), 0);
        assert_eq!(value.fetch_max(black_box($int::MAX), SeqCst), 0);
        assert_eq!(value.load(SeqCst), $int::MAX);

        assert_eq!(value.fetch_min(black_box(-1), SeqCst), $int::MAX);
        assert_eq!(value.load(SeqCst), -1);
        assert_eq!(value.fetch_min(black_box(0), SeqCst), -1);
        assert_eq!(value.load(SeqCst), -1);
    }};
}

fn main() {
    check_unsigned!(AtomicU8, u8);
    check_unsigned!(AtomicU16, u16);
    check_unsigned!(AtomicU32, u32);
    #[cfg(target_has_atomic = "64")]
    check_unsigned!(AtomicU64, u64);
    check_unsigned!(AtomicUsize, usize);

    check_signed!(AtomicI8, i8);
    check_signed!(AtomicI16, i16);
    check_signed!(AtomicI32, i32);
    #[cfg(target_has_atomic = "64")]
    check_signed!(AtomicI64, i64);
}
