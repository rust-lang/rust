macro_rules! nonzero_uint_module {
    ($T:ident) => {
        use core::num::NonZero;
        const MAX: NonZero<$T> = NonZero::<$T>::MAX;

        const fn nz(n: $T) -> NonZero<$T> {
            NonZero::new(n).unwrap()
        }

        test_runtime_and_compiletime! {
            fn test_strict_add() {
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_add(0), nz(1));
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_add(1), nz(2));
                assert_eq_const_safe!(NonZero<$T>: nz(57).strict_add(173), nz(230));
                assert_eq_const_safe!(NonZero<$T>: nz($T::MAX - 1).strict_add(1), MAX);
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_add($T::MAX - 1), MAX);
                assert_eq_const_safe!(NonZero<$T>: MAX.strict_add(0), MAX);
            }

            fn test_strict_mul() {
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_mul(nz(1)), nz(1));
                assert_eq_const_safe!(NonZero<$T>: MAX.strict_mul(nz(1)), MAX);
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_mul(MAX), MAX);
                assert_eq_const_safe!(NonZero<$T>: nz(123).strict_mul(nz(2)), nz(246));
                assert_eq_const_safe!(NonZero<$T>: nz(2).strict_mul(nz(123)), nz(246));
                assert_eq_const_safe!(NonZero<$T>: nz($T::MAX / 2).strict_mul(nz(2)), nz($T::MAX - 1));
                assert_eq_const_safe!(NonZero<$T>: nz(2).strict_mul(nz($T::MAX / 2)), nz($T::MAX - 1));
            }

            fn test_strict_pow() {
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_pow(0), nz(1));
                assert_eq_const_safe!(NonZero<$T>: nz(1).strict_pow(1), nz(1));
                assert_eq_const_safe!(NonZero<$T>: nz(3).strict_pow(0), nz(1));
                assert_eq_const_safe!(NonZero<$T>: nz(3).strict_pow(1), nz(3));
                assert_eq_const_safe!(NonZero<$T>: nz(3).strict_pow(2), nz(9));
                assert_eq_const_safe!(NonZero<$T>: MAX.strict_pow(0), nz(1));
                assert_eq_const_safe!(NonZero<$T>: MAX.strict_pow(1), MAX);
            }
        }

        #[test]
        #[should_panic = "attempt to add with overflow"]
        fn test_strict_add_overflow_big_small() {
            let _ = MAX.strict_add(1);
        }

        #[test]
        #[should_panic = "attempt to add with overflow"]
        fn test_strict_add_overflow_small_big() {
            let _ = nz(1).strict_add($T::MAX);
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
        #[should_panic = "attempt to exponentiate with overflow"]
        fn test_strict_pow_overflow_big_base() {
            let _ = MAX.strict_pow(2);
        }

        #[test]
        #[should_panic = "attempt to exponentiate with overflow"]
        fn test_strict_pow_overflow_big_exp() {
            let _ = nz(2).strict_pow(256);
        }
    };
}

mod u8 {
    nonzero_uint_module!(u8);
}

mod u16 {
    nonzero_uint_module!(u16);
}

mod u32 {
    nonzero_uint_module!(u32);
}

mod u64 {
    nonzero_uint_module!(u64);
}

mod u128 {
    nonzero_uint_module!(u128);
}

mod usize {
    nonzero_uint_module!(usize);
}
