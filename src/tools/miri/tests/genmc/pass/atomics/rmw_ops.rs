//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// This test check for correct handling of atomic read-modify-write operations for all integer sizes.
// Atomic max and min should return the previous value, and store the result in the atomic.
// Atomic addition and subtraction should have wrapping semantics.
// `and`, `nand`, `or`, `xor` should behave like their non-atomic counterparts.

// FIXME(genmc): add 128 bit atomics for platforms that support it, once GenMC gets 128 bit atomic support

#![no_main]

use std::sync::atomic::*;

const ORD: Ordering = Ordering::SeqCst;

fn assert_eq<T: Eq>(x: T, y: T) {
    if x != y {
        std::process::abort();
    }
}

macro_rules! test_rmw_edge_cases {
    ($int:ty, $atomic:ty) => {{
        // MAX, ADD
        let x = <$atomic>::new(123);
        assert_eq(123, x.fetch_max(0, ORD)); // `max` keeps existing value
        assert_eq(123, x.fetch_max(<$int>::MAX, ORD)); // `max` stores the new value
        assert_eq(<$int>::MAX, x.fetch_add(10, ORD)); // `fetch_add` should be wrapping
        assert_eq(<$int>::MAX.wrapping_add(10), x.load(ORD));

        // MIN, SUB
        x.store(42, ORD);
        assert_eq(42, x.fetch_min(<$int>::MAX, ORD)); // `min` keeps existing value
        assert_eq(42, x.fetch_min(<$int>::MIN, ORD)); // `min` stores the new value
        assert_eq(<$int>::MIN, x.fetch_sub(10, ORD)); // `fetch_sub` should be wrapping
        assert_eq(<$int>::MIN.wrapping_sub(10), x.load(ORD));

        // Small enough pattern to work for all integer sizes.
        let pattern = 0b01010101;

        // AND
        x.store(!0, ORD);
        assert_eq(!0, x.fetch_and(pattern, ORD));
        assert_eq(!0 & pattern, x.load(ORD));

        // NAND
        x.store(!0, ORD);
        assert_eq(!0, x.fetch_nand(pattern, ORD));
        assert_eq(!(!0 & pattern), x.load(ORD));

        // OR
        x.store(!0, ORD);
        assert_eq(!0, x.fetch_or(pattern, ORD));
        assert_eq(!0 | pattern, x.load(ORD));

        // XOR
        x.store(!0, ORD);
        assert_eq(!0, x.fetch_xor(pattern, ORD));
        assert_eq(!0 ^ pattern, x.load(ORD));

        // SWAP
        x.store(!0, ORD);
        assert_eq(!0, x.swap(pattern, ORD));
        assert_eq(pattern, x.load(ORD));

        // Check correct behavior of atomic min/max combined with overflowing add/sub.
        x.store(10, ORD);
        assert_eq(10, x.fetch_add(<$int>::MAX, ORD)); // definitely overflows, so new value of x is smaller than 10
        assert_eq(<$int>::MAX.wrapping_add(10), x.fetch_max(10, ORD)); // new value of x should be 10
        // assert_eq(10, x.load(ORD)); // FIXME(genmc,#4572): enable this check once GenMC correctly handles min/max truncation.
    }};
}

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    test_rmw_edge_cases!(u8, AtomicU8);
    test_rmw_edge_cases!(u16, AtomicU16);
    test_rmw_edge_cases!(u32, AtomicU32);
    test_rmw_edge_cases!(u64, AtomicU64);
    test_rmw_edge_cases!(usize, AtomicUsize);
    test_rmw_edge_cases!(i8, AtomicI8);
    test_rmw_edge_cases!(i16, AtomicI16);
    test_rmw_edge_cases!(i32, AtomicI32);
    test_rmw_edge_cases!(i64, AtomicI64);
    test_rmw_edge_cases!(isize, AtomicIsize);

    0
}
