// run-pass

#![feature(const_checked_int_methods)]
#![feature(const_euclidean_int_methods)]
#![feature(const_overflowing_int_methods)]
#![feature(const_wrapping_int_methods)]

macro_rules! suite {
    ($(
        $fn:ident -> $ty:ty { $( $label:ident : $expr:expr, $result:expr; )* }
    )*) => { $(
        fn $fn() {
            $(
                const $label: $ty = $expr;
                assert_eq!($label, $result);
            )*
        }
    )* }
}

suite!(
    checked -> Option<i8> {
        // `const_checked_int_methods`
        C1: 5i8.checked_add(2), Some(7);
        C2: 127i8.checked_add(2), None;

        C3: 5i8.checked_sub(2), Some(3);
        C4: (-127i8).checked_sub(2), None;

        C5: 1i8.checked_mul(3), Some(3);
        C6: 5i8.checked_mul(122), None;
        C7: (-127i8).checked_mul(-99), None;

        C8: (i8::MIN + 1).checked_div(-1), Some(127);
        C9: i8::MIN.checked_div(-1), None;
        C10: 1i8.checked_div(0), None;

        C11: 5i8.checked_rem(2), Some(1);
        C12: 5i8.checked_rem(0), None;
        C13: i8::MIN.checked_rem(-1), None;

        C14: 5i8.checked_neg(), Some(-5);
        C15: i8::MIN.checked_neg(), None;

        C16: 0x1i8.checked_shl(4), Some(0x10);
        C17: 0x1i8.checked_shl(129), None;

        C18: 0x10i8.checked_shr(4), Some(0x1);
        C19: 0x10i8.checked_shr(128), None;


        C20: (-5i8).checked_abs(), Some(5);
        C21: i8::MIN.checked_abs(), None;

        // `const_euclidean_int_methods`
        C22: (i8::MIN + 1).checked_div_euclid(-1), Some(127);
        C23: i8::MIN.checked_div_euclid(-1), None;
        C24: (1i8).checked_div_euclid(0), None;

        C25: 5i8.checked_rem_euclid(2), Some(1);
        C26: 5i8.checked_rem_euclid(0), None;
        C27: i8::MIN.checked_rem_euclid(-1), None;
    }
    checked_i128 -> Option<i128> {
        CHK_ADD_I128: i128::MAX.checked_add(1), None;
        CHK_MUL_I128: i128::MIN.checked_mul(-1), None;
    }

    saturating_and_wrapping -> i8 {
        // `const_saturating_int_methods`
        C28: 100i8.saturating_add(1), 101;
        C29: i8::MAX.saturating_add(100), i8::MAX;
        C30: i8::MIN.saturating_add(-1), i8::MIN;

        C31: 100i8.saturating_sub(127), -27;
        C32: i8::MIN.saturating_sub(100), i8::MIN;
        C33: i8::MAX.saturating_sub(-1), i8::MAX;

        C34: 10i8.saturating_mul(12), 120;
        C35: i8::MAX.saturating_mul(10), i8::MAX;
        C36: i8::MIN.saturating_mul(10), i8::MIN;

        C37: 100i8.saturating_neg(), -100;
        C38: (-100i8).saturating_neg(), 100;
        C39: i8::MIN.saturating_neg(), i8::MAX;
        C40: i8::MAX.saturating_neg(), i8::MIN + 1;

        C57: 100i8.saturating_abs(), 100;
        C58: (-100i8).saturating_abs(), 100;
        C59: i8::MIN.saturating_abs(), i8::MAX;
        C60: (i8::MIN + 1).saturating_abs(), i8::MAX;

        // `const_wrapping_int_methods`
        C41: 100i8.wrapping_div(10), 10;
        C42: (-128i8).wrapping_div(-1), -128;

        C43: 100i8.wrapping_rem(10), 0;
        C44: (-128i8).wrapping_rem(-1), 0;

        // `const_euclidean_int_methods`
        C45: 100i8.wrapping_div_euclid(10), 10;
        C46: (-128i8).wrapping_div_euclid(-1), -128;

        C47: 100i8.wrapping_rem_euclid(10), 0;
        C48: (-128i8).wrapping_rem_euclid(-1), 0;
    }
    saturating_and_wrapping_i128 -> i128 {
        SAT_ADD_I128: i128::MAX.saturating_add(1), i128::MAX;
        SAT_MUL_I128: i128::MAX.saturating_mul(2), i128::MAX;

        WRP_ADD_I128: i128::MAX.wrapping_add(1), i128::MIN;
        WRP_MUL_I128: i128::MAX.wrapping_mul(3), i128::MAX-2;
    }

    overflowing -> (i8, bool) {
        // `const_overflowing_int_methods`
        C49: 5i8.overflowing_div(2), (2, false);
        C50: i8::MIN.overflowing_div(-1), (i8::MIN, true);

        C51: 5i8.overflowing_rem(2), (1, false);
        C52: i8::MIN.overflowing_rem(-1), (0, true);

        // `const_euclidean_int_methods`
        C53: 5i8.overflowing_div_euclid(2), (2, false);
        C54: i8::MIN.overflowing_div_euclid(-1), (i8::MIN, true);

        C55: 5i8.overflowing_rem_euclid(2), (1, false);
        C56: i8::MIN.overflowing_rem_euclid(-1), (0, true);
    }
    overflowing_i128 -> (i128, bool) {
        OFL_ADD_I128: i128::MAX.overflowing_add(1), (i128::MIN, true);
        OFL_MUL_I128: i128::MAX.overflowing_mul(3), (i128::MAX-2, true);
    }
);

fn main() {
   checked();
   checked_i128();
   saturating_and_wrapping();
   saturating_and_wrapping_i128();
   overflowing();
   overflowing_i128();
}
