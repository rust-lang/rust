// Compiler:
//
// Run-time:
//   status: 0

#![feature(bench_black_box, core_intrinsics, start)]

#![no_std]

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    core::intrinsics::abort();
}

/*
 * Code
 */

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize {
    let one: isize = core::hint::black_box(1);

    macro_rules! check {
        ($ty:ty, $expr:expr) => {
            {
                const EXPECTED: $ty = {
                    #[allow(non_upper_case_globals)]
                    #[allow(dead_code)]
                    const one: isize = 1;
                    $expr
                };
                assert_eq!($expr, EXPECTED);
            }
        };
    }

    check!(u32, (2220326408_u32 + one as u32) >> (32 - 6));

    /// Generate `check!` tests for integer types at least as wide as 128 bits.
    macro_rules! check_ops128 {
        () => {
            check_ops64!();

            // Shifts.
            check!(T, VAL1 << (one + 63) as T);
            check!(T, VAL1 << (one + 80) as T);
            check!(T, VAL3 << (one + 62) as T);
            check!(T, VAL3 << (one + 63) as T);

            check!(T, VAL1 >> (one + 63) as T);
            check!(T, VAL2 >> (one + 63) as T);
            check!(T, VAL3 >> (one + 63) as T);
            check!(T, VAL3 >> (one + 80) as T);
        };
    }

    /// Generate `check!` tests for integer types at least as wide as 64 bits.
    macro_rules! check_ops64 {
        () => {
            check_ops32!();

            // Shifts.
            check!(T, VAL2 << (one + 32) as T);
            check!(T, VAL2 << (one + 48) as T);
            check!(T, VAL2 << (one + 60) as T);
            check!(T, VAL2 << (one + 62) as T);

            check!(T, VAL3 << (one + 32) as T);
            check!(T, VAL3 << (one + 48) as T);
            check!(T, VAL3 << (one + 60) as T);

            check!(T, VAL1 >> (one + 32) as T);
            check!(T, VAL1 >> (one + 48) as T);
            check!(T, VAL1 >> (one + 60) as T);
            check!(T, VAL1 >> (one + 62) as T);

            check!(T, VAL2 >> (one + 32) as T);
            check!(T, VAL2 >> (one + 48) as T);
            check!(T, VAL2 >> (one + 60) as T);
            check!(T, VAL2 >> (one + 62) as T);

            check!(T, VAL3 >> (one + 32) as T);
            check!(T, VAL3 >> (one + 48) as T);
            check!(T, VAL3 >> (one + 60) as T);
            check!(T, VAL3 >> (one + 62) as T);
        };
    }

    /// Generate `check!` tests for integer types at least as wide as 32 bits.
    macro_rules! check_ops32 {
        () => {
            // Shifts.
            check!(T, VAL2 << one as T);
            check!(T, VAL2 << (one as T - 1));

            check!(T, VAL3 << one as T);
            check!(T, VAL3 << (one as T - 1));

            check!(T, VAL1.wrapping_shl(one as u32 - 1));
            check!(T, VAL1.wrapping_shl(one as u32));
            check!(T, VAL1.wrapping_shl((one + 32) as u32));
            check!(T, VAL1.wrapping_shl((one + 48) as u32));
            check!(T, VAL1.wrapping_shl((one + 60) as u32));
            check!(T, VAL1.wrapping_shl((one + 62) as u32));
            check!(T, VAL1.wrapping_shl((one + 63) as u32));
            check!(T, VAL1.wrapping_shl((one + 80) as u32));

            check!(Option<T>, VAL1.checked_shl(one as u32 - 1));
            check!(Option<T>, VAL1.checked_shl(one as u32));
            check!(Option<T>, VAL1.checked_shl((one + 32) as u32));
            check!(Option<T>, VAL1.checked_shl((one + 48) as u32));
            check!(Option<T>, VAL1.checked_shl((one + 60) as u32));
            check!(Option<T>, VAL1.checked_shl((one + 62) as u32));
            check!(Option<T>, VAL1.checked_shl((one + 63) as u32));
            check!(Option<T>, VAL1.checked_shl((one + 80) as u32));

            check!(T, VAL1 >> (one as T - 1));
            check!(T, VAL1 >> one as T);

            check!(T, VAL2 >> one as T);
            check!(T, VAL2 >> (one as T - 1));

            check!(T, VAL3 >> (one as T - 1));
            check!(T, VAL3 >> one as T);

            check!(T, VAL1.wrapping_shr(one as u32 - 1));
            check!(T, VAL1.wrapping_shr(one as u32));
            check!(T, VAL1.wrapping_shr((one + 32) as u32));
            check!(T, VAL1.wrapping_shr((one + 48) as u32));
            check!(T, VAL1.wrapping_shr((one + 60) as u32));
            check!(T, VAL1.wrapping_shr((one + 62) as u32));
            check!(T, VAL1.wrapping_shr((one + 63) as u32));
            check!(T, VAL1.wrapping_shr((one + 80) as u32));

            check!(Option<T>, VAL1.checked_shr(one as u32 - 1));
            check!(Option<T>, VAL1.checked_shr(one as u32));
            check!(Option<T>, VAL1.checked_shr((one + 32) as u32));
            check!(Option<T>, VAL1.checked_shr((one + 48) as u32));
            check!(Option<T>, VAL1.checked_shr((one + 60) as u32));
            check!(Option<T>, VAL1.checked_shr((one + 62) as u32));
            check!(Option<T>, VAL1.checked_shr((one + 63) as u32));
            check!(Option<T>, VAL1.checked_shr((one + 80) as u32));

            // Casts
            check!(u64, (VAL1 >> one as T) as u64);

            // Addition.
            check!(T, VAL1 + one as T);
            check!(T, VAL2 + one as T);
            check!(T, VAL2 + (VAL2 + one as T) as T);
            check!(T, VAL3 + one as T);

            check!(Option<T>, VAL1.checked_add(one as T));
            check!(Option<T>, VAL2.checked_add(one as T));
            check!(Option<T>, VAL2.checked_add((VAL2 + one as T) as T));
            check!(Option<T>, VAL3.checked_add(T::MAX));
            check!(Option<T>, VAL3.checked_add(T::MIN));

            check!(T, VAL1.wrapping_add(one as T));
            check!(T, VAL2.wrapping_add(one as T));
            check!(T, VAL2.wrapping_add((VAL2 + one as T) as T));
            check!(T, VAL3.wrapping_add(T::MAX));
            check!(T, VAL3.wrapping_add(T::MIN));

            check!((T, bool), VAL1.overflowing_add(one as T));
            check!((T, bool), VAL2.overflowing_add(one as T));
            check!((T, bool), VAL2.overflowing_add((VAL2 + one as T) as T));
            check!((T, bool), VAL3.overflowing_add(T::MAX));
            check!((T, bool), VAL3.overflowing_add(T::MIN));

            check!(T, VAL1.saturating_add(one as T));
            check!(T, VAL2.saturating_add(one as T));
            check!(T, VAL2.saturating_add((VAL2 + one as T) as T));
            check!(T, VAL3.saturating_add(T::MAX));
            check!(T, VAL3.saturating_add(T::MIN));

            // Subtraction
            check!(T, VAL1 - one as T);
            check!(T, VAL2 - one as T);
            check!(T, VAL3 - one as T);

            check!(Option<T>, VAL1.checked_sub(one as T));
            check!(Option<T>, VAL2.checked_sub(one as T));
            check!(Option<T>, VAL2.checked_sub((VAL2 + one as T) as T));
            check!(Option<T>, VAL3.checked_sub(T::MAX));
            check!(Option<T>, VAL3.checked_sub(T::MIN));

            check!(T, VAL1.wrapping_sub(one as T));
            check!(T, VAL2.wrapping_sub(one as T));
            check!(T, VAL2.wrapping_sub((VAL2 + one as T) as T));
            check!(T, VAL3.wrapping_sub(T::MAX));
            check!(T, VAL3.wrapping_sub(T::MIN));

            check!((T, bool), VAL1.overflowing_sub(one as T));
            check!((T, bool), VAL2.overflowing_sub(one as T));
            check!((T, bool), VAL2.overflowing_sub((VAL2 + one as T) as T));
            check!((T, bool), VAL3.overflowing_sub(T::MAX));
            check!((T, bool), VAL3.overflowing_sub(T::MIN));

            check!(T, VAL1.saturating_sub(one as T));
            check!(T, VAL2.saturating_sub(one as T));
            check!(T, VAL2.saturating_sub((VAL2 + one as T) as T));
            check!(T, VAL3.saturating_sub(T::MAX));
            check!(T, VAL3.saturating_sub(T::MIN));

            // Multiplication
            check!(T, VAL1 * (one + 1) as T);
            check!(T, VAL1 * (one as T + VAL2));
            check!(T, VAL2 * (one + 1) as T);
            check!(T, VAL2 * (one as T + VAL2));
            check!(T, VAL3 * one as T);
            check!(T, VAL4 * (one + 1) as T);
            check!(T, VAL5 * (one + 1) as T);

            check!(Option<T>, VAL1.checked_mul((one + 1) as T));
            check!(Option<T>, VAL1.checked_mul((one as T + VAL2)));
            check!(Option<T>, VAL3.checked_mul(VAL3));
            check!(Option<T>, VAL4.checked_mul((one + 1) as T));
            check!(Option<T>, VAL5.checked_mul((one + 1) as T));

            check!(T, VAL1.wrapping_mul((one + 1) as T));
            check!(T, VAL1.wrapping_mul((one as T + VAL2)));
            check!(T, VAL3.wrapping_mul(VAL3));
            check!(T, VAL4.wrapping_mul((one + 1) as T));
            check!(T, VAL5.wrapping_mul((one + 1) as T));

            check!((T, bool), VAL1.overflowing_mul((one + 1) as T));
            check!((T, bool), VAL1.overflowing_mul((one as T + VAL2)));
            check!((T, bool), VAL3.overflowing_mul(VAL3));
            check!((T, bool), VAL4.overflowing_mul((one + 1) as T));
            check!((T, bool), VAL5.overflowing_mul((one + 1) as T));

            check!(T, VAL1.saturating_mul((one + 1) as T));
            check!(T, VAL1.saturating_mul((one as T + VAL2)));
            check!(T, VAL3.saturating_mul(VAL3));
            check!(T, VAL4.saturating_mul((one + 1) as T));
            check!(T, VAL5.saturating_mul((one + 1) as T));

            // Division.
            check!(T, VAL1 / (one + 1) as T);
            check!(T, VAL1 / (one + 2) as T);

            check!(T, VAL2 / (one + 1) as T);
            check!(T, VAL2 / (one + 2) as T);

            check!(T, VAL3 / (one + 1) as T);
            check!(T, VAL3 / (one + 2) as T);
            check!(T, VAL3 / (one as T + VAL4));
            check!(T, VAL3 / (one as T + VAL2));

            check!(T, VAL4 / (one + 1) as T);
            check!(T, VAL4 / (one + 2) as T);

            check!(Option<T>, VAL1.checked_div((one + 1) as T));
            check!(Option<T>, VAL1.checked_div((one as T + VAL2)));
            check!(Option<T>, VAL3.checked_div(VAL3));
            check!(Option<T>, VAL4.checked_div((one + 1) as T));
            check!(Option<T>, VAL5.checked_div((one + 1) as T));
            check!(Option<T>, (T::MIN).checked_div((0 as T).wrapping_sub(one as T)));
            check!(Option<T>, VAL5.checked_div((one - 1) as T)); // var5 / 0

            check!(T, VAL1.wrapping_div((one + 1) as T));
            check!(T, VAL1.wrapping_div((one as T + VAL2)));
            check!(T, VAL3.wrapping_div(VAL3));
            check!(T, VAL4.wrapping_div((one + 1) as T));
            check!(T, VAL5.wrapping_div((one + 1) as T));
            check!(T, (T::MIN).wrapping_div((0 as T).wrapping_sub(one as T)));

            check!((T, bool), VAL1.overflowing_div((one + 1) as T));
            check!((T, bool), VAL1.overflowing_div((one as T + VAL2)));
            check!((T, bool), VAL3.overflowing_div(VAL3));
            check!((T, bool), VAL4.overflowing_div((one + 1) as T));
            check!((T, bool), VAL5.overflowing_div((one + 1) as T));
            check!((T, bool), (T::MIN).overflowing_div((0 as T).wrapping_sub(one as T)));

            check!(T, VAL1.saturating_div((one + 1) as T));
            check!(T, VAL1.saturating_div((one as T + VAL2)));
            check!(T, VAL3.saturating_div(VAL3));
            check!(T, VAL4.saturating_div((one + 1) as T));
            check!(T, VAL5.saturating_div((one + 1) as T));
            check!(T, (T::MIN).saturating_div((0 as T).wrapping_sub(one as T)));
        };
    }

    {
        type T = u32;
        const VAL1: T = 14162_u32;
        const VAL2: T = 14556_u32;
        const VAL3: T = 323656954_u32;
        const VAL4: T = 2023651954_u32;
        const VAL5: T = 1323651954_u32;
        check_ops32!();
    }

    {
        type T = i32;
        const VAL1: T = 13456_i32;
        const VAL2: T = 10475_i32;
        const VAL3: T = 923653954_i32;
        const VAL4: T = 993198738_i32;
        const VAL5: T = 1023653954_i32;
        check_ops32!();
    }

    {
        type T = u64;
        const VAL1: T = 134217856_u64;
        const VAL2: T = 104753732_u64;
        const VAL3: T = 12323651988970863954_u64;
        const VAL4: T = 7323651988970863954_u64;
        const VAL5: T = 8323651988970863954_u64;
        check_ops64!();
    }

    {
        type T = i64;
        const VAL1: T = 134217856_i64;
        const VAL2: T = 104753732_i64;
        const VAL3: T = 6323651988970863954_i64;
        const VAL4: T = 2323651988970863954_i64;
        const VAL5: T = 3323651988970863954_i64;
        check_ops64!();
    }

    {
        type T = u128;
        const VAL1: T = 134217856_u128;
        const VAL2: T = 10475372733397991552_u128;
        const VAL3: T = 193236519889708027473620326106273939584_u128;
        const VAL4: T = 123236519889708027473620326106273939584_u128;
        const VAL5: T = 153236519889708027473620326106273939584_u128;
        check_ops128!();
    }
    {
        type T = i128;
        const VAL1: T = 134217856_i128;
        const VAL2: T = 10475372733397991552_i128;
        const VAL3: T = 83236519889708027473620326106273939584_i128;
        const VAL4: T = 63236519889708027473620326106273939584_i128;
        const VAL5: T = 73236519889708027473620326106273939584_i128;
        check_ops128!();
    }

    0
}
