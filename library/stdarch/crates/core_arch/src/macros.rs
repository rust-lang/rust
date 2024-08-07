//! Utility macros.

#[allow(unused)]
macro_rules! static_assert {
    ($e:expr) => {
        const {
            assert!($e);
        }
    };
    ($e:expr, $msg:expr) => {
        const {
            assert!($e, $msg);
        }
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_uimm_bits {
    ($imm:ident, $bits:expr) => {
        // `0 <= $imm` produces a warning if the immediate has an unsigned type
        #[allow(unused_comparisons)]
        {
            static_assert!(
                0 <= $imm && $imm < (1 << $bits),
                concat!(
                    stringify!($imm),
                    " doesn't fit in ",
                    stringify!($bits),
                    " bits",
                )
            )
        }
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_simm_bits {
    ($imm:ident, $bits:expr) => {
        static_assert!(
            (-1 << ($bits - 1)) - 1 <= $imm && $imm < (1 << ($bits - 1)),
            concat!(
                stringify!($imm),
                " doesn't fit in ",
                stringify!($bits),
                " bits",
            )
        )
    };
}

#[allow(unused)]
macro_rules! types {
    ($(
        $(#[$doc:meta])*
        pub struct $name:ident($len:literal x $v:vis $elem_type:ty);
    )*) => ($(
        $(#[$doc])*
        #[derive(Copy, Clone, Debug)]
        #[allow(non_camel_case_types)]
        #[repr(simd)]
        #[allow(clippy::missing_inline_in_public_items)]
        pub struct $name($v [$elem_type; $len]);

        impl $name {
            /// Using `my_simd([x; N])` seemingly fails tests,
            /// so use this internal helper for it instead.
            #[inline(always)]
            $v fn splat(value: $elem_type) -> $name {
                #[derive(Copy, Clone)]
                #[repr(simd)]
                struct JustOne([$elem_type; 1]);
                let one = JustOne([value]);
                // SAFETY: 0 is always in-bounds because we're shuffling
                // a simd type with exactly one element.
                unsafe { simd_shuffle!(one, one, [0; $len]) }
            }
        }
    )*)
}

#[allow(unused)]
macro_rules! simd_shuffle {
    ($x:expr, $y:expr, $idx:expr $(,)?) => {{
        simd_shuffle::<_, [u32; _], _>($x, $y, const { $idx })
    }};
}

#[allow(unused)]
macro_rules! simd_insert {
    ($x:expr, $idx:expr, $val:expr $(,)?) => {{
        simd_insert($x, const { $idx }, $val)
    }};
}

#[allow(unused)]
macro_rules! simd_extract {
    ($x:expr, $idx:expr $(,)?) => {{
        simd_extract($x, const { $idx })
    }};
    ($x:expr, $idx:expr, $ty:ty $(,)?) => {{
        simd_extract::<_, $ty>($x, const { $idx })
    }};
}
