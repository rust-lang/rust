// Compiler:
//
// Run-time:
//   status: 0

#![feature(const_black_box)]

fn main() {
    use std::hint::black_box;

    macro_rules! check {
        ($ty:ty, $expr:expr) => {
            {
                const EXPECTED: $ty = $expr;
                assert_eq!($expr, EXPECTED);
            }
        };
    }

    check!(u32, (2220326408_u32 + black_box(1)) >> (32 - 6));

    /// Generate `check!` tests for integer types at least as wide as 128 bits.
    macro_rules! check_ops128 {
        () => {
            check_ops64!();

            // Shifts.
            check!(T, VAL1 << black_box(64));
            check!(T, VAL1 << black_box(81));
            check!(T, VAL3 << black_box(63));
            check!(T, VAL3 << black_box(64));

            check!(T, VAL1 >> black_box(64));
            check!(T, VAL2 >> black_box(64));
            check!(T, VAL3 >> black_box(64));
            check!(T, VAL3 >> black_box(81));
        };
    }

    /// Generate `check!` tests for integer types at least as wide as 64 bits.
    macro_rules! check_ops64 {
        () => {
            check_ops32!();

            // Shifts.
            check!(T, VAL2 << black_box(33));
            check!(T, VAL2 << black_box(49));
            check!(T, VAL2 << black_box(61));
            check!(T, VAL2 << black_box(63));

            check!(T, VAL3 << black_box(33));
            check!(T, VAL3 << black_box(49));
            check!(T, VAL3 << black_box(61));

            check!(T, VAL1 >> black_box(33));
            check!(T, VAL1 >> black_box(49));
            check!(T, VAL1 >> black_box(61));
            check!(T, VAL1 >> black_box(63));

            check!(T, VAL2 >> black_box(33));
            check!(T, VAL2 >> black_box(49));
            check!(T, VAL2 >> black_box(61));
            check!(T, VAL2 >> black_box(63));

            check!(T, VAL3 >> black_box(33));
            check!(T, VAL3 >> black_box(49));
            check!(T, VAL3 >> black_box(61));
            check!(T, VAL3 >> black_box(63));
        };
    }

    /// Generate `check!` tests for integer types at least as wide as 32 bits.
    macro_rules! check_ops32 {
        () => {
            // Shifts.
            check!(T, VAL2 << black_box(1));
            check!(T, VAL2 << black_box(0));

            check!(T, VAL3 << black_box(1));
            check!(T, VAL3 << black_box(0));

            check!(T, VAL1.wrapping_shl(black_box(0)));
            check!(T, VAL1.wrapping_shl(black_box(1)));
            check!(T, VAL1.wrapping_shl(black_box(33)));
            check!(T, VAL1.wrapping_shl(black_box(49)));
            check!(T, VAL1.wrapping_shl(black_box(61)));
            check!(T, VAL1.wrapping_shl(black_box(63)));
            check!(T, VAL1.wrapping_shl(black_box(64)));
            check!(T, VAL1.wrapping_shl(black_box(81)));

            check!(Option<T>, VAL1.checked_shl(black_box(0)));
            check!(Option<T>, VAL1.checked_shl(black_box(1)));
            check!(Option<T>, VAL1.checked_shl(black_box(33)));
            check!(Option<T>, VAL1.checked_shl(black_box(49)));
            check!(Option<T>, VAL1.checked_shl(black_box(61)));
            check!(Option<T>, VAL1.checked_shl(black_box(63)));
            check!(Option<T>, VAL1.checked_shl(black_box(64)));
            check!(Option<T>, VAL1.checked_shl(black_box(81)));

            check!(T, VAL1 >> black_box(0));
            check!(T, VAL1 >> black_box(1));

            check!(T, VAL2 >> black_box(1));
            check!(T, VAL2 >> black_box(0));

            check!(T, VAL3 >> black_box(0));
            check!(T, VAL3 >> black_box(1));

            check!(T, VAL1.wrapping_shr(black_box(0)));
            check!(T, VAL1.wrapping_shr(black_box(1)));
            check!(T, VAL1.wrapping_shr(black_box(33)));
            check!(T, VAL1.wrapping_shr(black_box(49)));
            check!(T, VAL1.wrapping_shr(black_box(61)));
            check!(T, VAL1.wrapping_shr(black_box(63)));
            check!(T, VAL1.wrapping_shr(black_box(64)));
            check!(T, VAL1.wrapping_shr(black_box(81)));

            check!(Option<T>, VAL1.checked_shr(black_box(0)));
            check!(Option<T>, VAL1.checked_shr(black_box(1)));
            check!(Option<T>, VAL1.checked_shr(black_box(33)));
            check!(Option<T>, VAL1.checked_shr(black_box(49)));
            check!(Option<T>, VAL1.checked_shr(black_box(61)));
            check!(Option<T>, VAL1.checked_shr(black_box(63)));
            check!(Option<T>, VAL1.checked_shr(black_box(64)));
            check!(Option<T>, VAL1.checked_shr(black_box(81)));

            // Casts
            check!(u64, (VAL1 >> black_box(1)) as u64);

            // Addition.
            check!(T, VAL1 + black_box(1));
            check!(T, VAL2 + black_box(1));
            check!(T, VAL2 + (VAL2 + black_box(1)));
            check!(T, VAL3 + black_box(1));

            check!(Option<T>, VAL1.checked_add(black_box(1)));
            check!(Option<T>, VAL2.checked_add(black_box(1)));
            check!(Option<T>, VAL2.checked_add(VAL2 + black_box(1)));
            check!(Option<T>, VAL3.checked_add(T::MAX));
            check!(Option<T>, VAL3.checked_add(T::MIN));

            check!(T, VAL1.wrapping_add(black_box(1)));
            check!(T, VAL2.wrapping_add(black_box(1)));
            check!(T, VAL2.wrapping_add(VAL2 + black_box(1)));
            check!(T, VAL3.wrapping_add(T::MAX));
            check!(T, VAL3.wrapping_add(T::MIN));

            check!((T, bool), VAL1.overflowing_add(black_box(1)));
            check!((T, bool), VAL2.overflowing_add(black_box(1)));
            check!((T, bool), VAL2.overflowing_add(VAL2 + black_box(1)));
            check!((T, bool), VAL3.overflowing_add(T::MAX));
            check!((T, bool), VAL3.overflowing_add(T::MIN));

            check!(T, VAL1.saturating_add(black_box(1)));
            check!(T, VAL2.saturating_add(black_box(1)));
            check!(T, VAL2.saturating_add(VAL2 + black_box(1)));
            check!(T, VAL3.saturating_add(T::MAX));
            check!(T, VAL3.saturating_add(T::MIN));

            // Subtraction
            check!(T, VAL1 - black_box(1));
            check!(T, VAL2 - black_box(1));
            check!(T, VAL3 - black_box(1));

            check!(Option<T>, VAL1.checked_sub(black_box(1)));
            check!(Option<T>, VAL2.checked_sub(black_box(1)));
            check!(Option<T>, VAL2.checked_sub(VAL2 + black_box(1)));
            check!(Option<T>, VAL3.checked_sub(T::MAX));
            check!(Option<T>, VAL3.checked_sub(T::MIN));

            check!(T, VAL1.wrapping_sub(black_box(1)));
            check!(T, VAL2.wrapping_sub(black_box(1)));
            check!(T, VAL2.wrapping_sub(VAL2 + black_box(1)));
            check!(T, VAL3.wrapping_sub(T::MAX));
            check!(T, VAL3.wrapping_sub(T::MIN));

            check!((T, bool), VAL1.overflowing_sub(black_box(1)));
            check!((T, bool), VAL2.overflowing_sub(black_box(1)));
            check!((T, bool), VAL2.overflowing_sub(VAL2 + black_box(1)));
            check!((T, bool), VAL3.overflowing_sub(T::MAX));
            check!((T, bool), VAL3.overflowing_sub(T::MIN));

            check!(T, VAL1.saturating_sub(black_box(1)));
            check!(T, VAL2.saturating_sub(black_box(1)));
            check!(T, VAL2.saturating_sub(VAL2 + black_box(1)));
            check!(T, VAL3.saturating_sub(T::MAX));
            check!(T, VAL3.saturating_sub(T::MIN));

            // Multiplication
            check!(T, VAL1 * black_box(2));
            check!(T, VAL1 * (black_box(1) + VAL2));
            check!(T, VAL2 * black_box(2));
            check!(T, VAL2 * (black_box(1) + VAL2));
            check!(T, VAL3 * black_box(1));
            check!(T, VAL4 * black_box(2));
            check!(T, VAL5 * black_box(2));

            check!(Option<T>, VAL1.checked_mul(black_box(2)));
            check!(Option<T>, VAL1.checked_mul(black_box(1) + VAL2));
            check!(Option<T>, VAL3.checked_mul(VAL3));
            check!(Option<T>, VAL4.checked_mul(black_box(2)));
            check!(Option<T>, VAL5.checked_mul(black_box(2)));

            check!(T, VAL1.wrapping_mul(black_box(2)));
            check!(T, VAL1.wrapping_mul((black_box(1) + VAL2)));
            check!(T, VAL3.wrapping_mul(VAL3));
            check!(T, VAL4.wrapping_mul(black_box(2)));
            check!(T, VAL5.wrapping_mul(black_box(2)));

            check!((T, bool), VAL1.overflowing_mul(black_box(2)));
            check!((T, bool), VAL1.overflowing_mul(black_box(1) + VAL2));
            check!((T, bool), VAL3.overflowing_mul(VAL3));
            check!((T, bool), VAL4.overflowing_mul(black_box(2)));
            check!((T, bool), VAL5.overflowing_mul(black_box(2)));

            check!(T, VAL1.saturating_mul(black_box(2)));
            check!(T, VAL1.saturating_mul(black_box(1) + VAL2));
            check!(T, VAL3.saturating_mul(VAL3));
            check!(T, VAL4.saturating_mul(black_box(2)));
            check!(T, VAL5.saturating_mul(black_box(2)));

            // Division.
            check!(T, VAL1 / black_box(2));
            check!(T, VAL1 / black_box(3));

            check!(T, VAL2 / black_box(2));
            check!(T, VAL2 / black_box(3));

            check!(T, VAL3 / black_box(2));
            check!(T, VAL3 / black_box(3));
            check!(T, VAL3 / (black_box(1) + VAL4));
            check!(T, VAL3 / (black_box(1) + VAL2));

            check!(T, VAL4 / black_box(2));
            check!(T, VAL4 / black_box(3));

            check!(Option<T>, VAL1.checked_div(black_box(2)));
            check!(Option<T>, VAL1.checked_div(black_box(1) + VAL2));
            check!(Option<T>, VAL3.checked_div(VAL3));
            check!(Option<T>, VAL4.checked_div(black_box(2)));
            check!(Option<T>, VAL5.checked_div(black_box(2)));
            check!(Option<T>, (T::MIN).checked_div(black_box(0 as T).wrapping_sub(1)));
            check!(Option<T>, VAL5.checked_div(black_box(0))); // var5 / 0

            check!(T, VAL1.wrapping_div(black_box(2)));
            check!(T, VAL1.wrapping_div(black_box(1) + VAL2));
            check!(T, VAL3.wrapping_div(VAL3));
            check!(T, VAL4.wrapping_div(black_box(2)));
            check!(T, VAL5.wrapping_div(black_box(2)));
            check!(T, (T::MIN).wrapping_div(black_box(0 as T).wrapping_sub(1)));

            check!((T, bool), VAL1.overflowing_div(black_box(2)));
            check!((T, bool), VAL1.overflowing_div(black_box(1) + VAL2));
            check!((T, bool), VAL3.overflowing_div(VAL3));
            check!((T, bool), VAL4.overflowing_div(black_box(2)));
            check!((T, bool), VAL5.overflowing_div(black_box(2)));
            check!((T, bool), (T::MIN).overflowing_div(black_box(0 as T).wrapping_sub(1)));

            check!(T, VAL1.saturating_div(black_box(2)));
            check!(T, VAL1.saturating_div((black_box(1) + VAL2)));
            check!(T, VAL3.saturating_div(VAL3));
            check!(T, VAL4.saturating_div(black_box(2)));
            check!(T, VAL5.saturating_div(black_box(2)));
            check!(T, (T::MIN).saturating_div((0 as T).wrapping_sub(black_box(1))));
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
}
