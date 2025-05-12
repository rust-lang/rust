//@ run-pass
//@ compile-flags: -C debug-assertions

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

// Check that we do *not* overflow on a number of edge cases.
// (compare with test/run-fail/overflowing-{lsh,rsh}*.rs)

fn main() {
    test_left_shift();
    test_right_shift();
}

pub static mut HACK: i32 = 0;

// Work around constant-evaluation
// The point of this test is to exercise the code generated for execution at runtime,
// `id` can never be flagged as a const fn by future aggressive analyses...
// due to the modification of the static
#[inline(never)]
fn id<T>(x: T) -> T {
    unsafe { HACK += 1; }
    x
}

fn test_left_shift() {
    // negative rhs can panic, but values in [0,N-1] are okay for iN

    macro_rules! tests {
        ($iN:ty, $uN:ty, $max_rhs:expr, $expect_i:expr, $expect_u:expr) => { {
            let x = (1 as $iN) << id(0);
            assert_eq!(x, 1);
            let x = (1 as $uN) << id(0);
            assert_eq!(x, 1);
            let x = (1 as $iN) << id($max_rhs);
            assert_eq!(x, $expect_i);
            let x = (1 as $uN) << id($max_rhs);
            assert_eq!(x, $expect_u);
            // high-order bits on LHS are silently discarded without panic.
            let x = (3 as $iN) << id($max_rhs);
            assert_eq!(x, $expect_i);
            let x = (3 as $uN) << id($max_rhs);
            assert_eq!(x, $expect_u);
        } }
    }

    let x = 1_i8 << id(0);
    assert_eq!(x, 1);
    let x = 1_u8 << id(0);
    assert_eq!(x, 1);
    let x = 1_i8 << id(7);
    assert_eq!(x, i8::MIN);
    let x = 1_u8 << id(7);
    assert_eq!(x, 0x80);
    // high-order bits on LHS are silently discarded without panic.
    let x = 3_i8 << id(7);
    assert_eq!(x, i8::MIN);
    let x = 3_u8 << id(7);
    assert_eq!(x, 0x80);

    // above is (approximately) expanded from:
    tests!(i8, u8, 7, i8::MIN, 0x80_u8);

    tests!(i16, u16, 15, i16::MIN, 0x8000_u16);
    tests!(i32, u32, 31, i32::MIN, 0x8000_0000_u32);
    tests!(i64, u64, 63, i64::MIN, 0x8000_0000_0000_0000_u64);
}

fn test_right_shift() {
    // negative rhs can panic, but values in [0,N-1] are okay for iN

    macro_rules! tests {
        ($iN:ty, $uN:ty, $max_rhs:expr,
         $signbit_i:expr, $highbit_i:expr, $highbit_u:expr) =>
        { {
            let x = (1 as $iN) >> id(0);
            assert_eq!(x, 1);
            let x = (1 as $uN) >> id(0);
            assert_eq!(x, 1);
            let x = ($highbit_i) >> id($max_rhs-1);
            assert_eq!(x, 1);
            let x = ($highbit_u) >> id($max_rhs);
            assert_eq!(x, 1);
            // sign-bit is carried by arithmetic right shift
            let x = ($signbit_i) >> id($max_rhs);
            assert_eq!(x, -1);
            // low-order bits on LHS are silently discarded without panic.
            let x = ($highbit_i + 1) >> id($max_rhs-1);
            assert_eq!(x, 1);
            let x = ($highbit_u + 1) >> id($max_rhs);
            assert_eq!(x, 1);
            let x = ($signbit_i + 1) >> id($max_rhs);
            assert_eq!(x, -1);
        } }
    }

    tests!(i8, u8, 7, i8::MIN, 0x40_i8, 0x80_u8);
    tests!(i16, u16, 15, i16::MIN, 0x4000_u16, 0x8000_u16);
    tests!(i32, u32, 31, i32::MIN, 0x4000_0000_u32, 0x8000_0000_u32);
    tests!(i64, u64, 63, i64::MIN,
           0x4000_0000_0000_0000_u64, 0x8000_0000_0000_0000_u64);
}
