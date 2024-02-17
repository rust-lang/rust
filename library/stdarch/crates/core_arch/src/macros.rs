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
        pub struct $name:ident($($fields:tt)*);
    )*) => ($(
        $(#[$doc])*
        #[derive(Copy, Clone, Debug)]
        #[allow(non_camel_case_types)]
        #[repr(simd)]
        #[allow(clippy::missing_inline_in_public_items)]
        pub struct $name($($fields)*);
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
