macro_rules! nonzero_int_module {
    ($T:ident) => {
        use core::num::NonZero;
        const MIN: NonZero<$T> = NonZero::<$T>::MIN;
        const MAX: NonZero<$T> = NonZero::<$T>::MAX;

        const fn nz(n: $T) -> NonZero<$T> {
            NonZero::new(n).unwrap()
        }

        test_runtime_and_compiletime! {
            fn test_strict_neg() {
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_neg(), nz(-1));
                assert_eq_const_safe!(NonZero<$T>: nz(-1).strict_neg(), nz(1));
                assert_eq_const_safe!(NonZero<$T>: nz(123).strict_neg(), nz(-123));
                assert_eq_const_safe!(NonZero<$T>: nz(-123).strict_neg(), nz(123));
                assert_eq_const_safe!(NonZero<$T>: nz($T::MAX - 2).strict_neg(), nz($T::MIN + 3));
                assert_eq_const_safe!(NonZero<$T>: nz($T::MAX - 1).strict_neg(), nz($T::MIN + 2));
                assert_eq_const_safe!(NonZero<$T>: MAX.strict_neg(), nz($T::MIN + 1));
                assert_eq_const_safe!(NonZero<$T>: nz($T::MIN + 2).strict_neg(), nz($T::MAX - 1));
                assert_eq_const_safe!(NonZero<$T>: nz($T::MIN + 1).strict_neg(), MAX);
            }

            fn test_strict_mul() {
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_mul(nz(1)), nz(1));
                assert_eq_const_safe!(NonZero<$T>: nz(-1).strict_mul(nz(1)), nz(-1));
                assert_eq_const_safe!(NonZero<$T>: nz(63).strict_mul(nz(-2)), nz(-126));
                assert_eq_const_safe!(NonZero<$T>: nz(-63).strict_mul(nz(-2)), nz(126));
                assert_eq_const_safe!(NonZero<$T>: MAX.strict_mul(nz(1)), MAX);
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_mul(MAX), MAX);
                assert_eq_const_safe!(NonZero<$T>: MIN.strict_mul(nz(1)), MIN);
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_mul(MIN), MIN);
                assert_eq_const_safe!(NonZero<$T>: MAX.strict_mul(nz(-1)), nz($T::MIN + 1));
                assert_eq_const_safe!(NonZero<$T>: nz(-1).strict_mul(MAX), nz($T::MIN + 1));
                assert_eq_const_safe!(NonZero<$T>: nz($T::MIN / 2).strict_mul(nz(2)), MIN);
                assert_eq_const_safe!(NonZero<$T>: nz(2).strict_mul(nz($T::MIN / 2)), MIN);
                assert_eq_const_safe!(NonZero<$T>: nz($T::MAX / 2).strict_mul(nz(2)), nz($T::MAX - 1));
                assert_eq_const_safe!(NonZero<$T>: nz(2).strict_mul(nz($T::MAX / 2)), nz($T::MAX - 1));
            }

            fn test_strict_pow() {
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_pow(0), nz(1));
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_pow(1), nz(1));
                assert_eq_const_safe!(NonZero<$T>: nz(-1).strict_pow(0), nz(1));
                assert_eq_const_safe!(NonZero<$T>: nz(-1).strict_pow(1), nz(-1));
                assert_eq_const_safe!(NonZero<$T>: nz(3).strict_pow(0), nz(1));
                assert_eq_const_safe!(NonZero<$T>: nz(3).strict_pow(1), nz(3));
                assert_eq_const_safe!(NonZero<$T>: nz(3).strict_pow(2), nz(9));
                assert_eq_const_safe!(NonZero<$T>: nz(-3).strict_pow(0), nz(1));
                assert_eq_const_safe!(NonZero<$T>: nz(-3).strict_pow(1), nz(-3));
                assert_eq_const_safe!(NonZero<$T>: nz(-3).strict_pow(2), nz(9));
                assert_eq_const_safe!(NonZero<$T>: MAX.strict_pow(0), nz(1));
                assert_eq_const_safe!(NonZero<$T>: MAX.strict_pow(1), MAX);
                assert_eq_const_safe!(NonZero<$T>: MIN.strict_pow(0), nz(1));
                assert_eq_const_safe!(NonZero<$T>: MIN.strict_pow(1), MIN);
            }
        }

        #[test]
        #[should_panic = "attempt to negate with overflow"]
        fn test_strict_neg_overflow() {
            let _ = MIN.strict_neg();
        }

        #[test]
        #[should_panic = "attempt to multiply with overflow"]
        fn test_strict_mul_overflow_neg_min() {
            let _ = MIN.strict_mul(nz(-1));
        }

        #[test]
        #[should_panic = "attempt to multiply with overflow"]
        fn test_strict_mul_overflow_neg_min_reverse() {
            let _ = nz(-1).strict_mul(MIN);
        }

        #[test]
        #[should_panic = "attempt to multiply with overflow"]
        fn test_strict_mul_overflow_limit() {
            let _ = nz($T::MAX / 2 + 1).strict_mul(nz(2));
        }

        #[test]
        #[should_panic = "attempt to multiply with overflow"]
        fn test_strict_mul_overflow_large() {
            let _ = MAX.strict_mul(nz(2));
        }

        #[test]
        #[should_panic = "attempt to multiply with overflow"]
        fn test_strict_mul_overflow_neg_large() {
            let _ = MIN.strict_mul(nz(2));
        }

        #[test]
        #[should_panic = "attempt to exponentiate with overflow"]
        fn test_strict_pow_overflow_big_base() {
            let _ = MAX.strict_pow(2);
        }

        #[test]
        #[should_panic = "attempt to exponentiate with overflow"]
        fn test_strict_pow_overflow_big_exp() {
            let _ = nz(2).strict_pow(256);
        }

        #[test]
        #[should_panic = "attempt to exponentiate with overflow"]
        fn test_strict_pow_overflow_neg_base() {
            let _ = MIN.strict_pow(2);
        }
    };
}

mod i8 {
    nonzero_int_module!(i8);
}

mod i16 {
    nonzero_int_module!(i16);
}

mod i32 {
    nonzero_int_module!(i32);
}

mod i64 {
    nonzero_int_module!(i64);
}

mod i128 {
    nonzero_int_module!(i128);
}

mod isize {
    nonzero_int_module!(isize);
}
