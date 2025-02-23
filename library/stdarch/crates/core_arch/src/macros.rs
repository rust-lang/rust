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
    (
        #![$stability_first:meta]
        $(
            #![$stability_more:meta]
        )*

        $(
            $(#[$doc:meta])*
            $(stability: [$stability_already: meta])*
            pub struct $name:ident($len:literal x $v:vis $elem_type:ty);
        )*
    ) => (types! {
        $(
            #![$stability_more]
        )*

        $(
            $(#[$doc])*
            $(stability: [$stability_already])*
            stability: [$stability_first]
            pub struct $name($len x $v $elem_type);
        )*
    });

    (
        $(
            $(#[$doc:meta])*
            $(stability: [$stability: meta])+
            pub struct $name:ident($len:literal x $v:vis $elem_type:ty);
        )*
    ) => ($(
        $(#[$doc])*
        $(#[$stability])+
        #[derive(Copy, Clone)]
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

            /// Returns an array reference containing the entire SIMD vector.
            $v const fn as_array(&self) -> &[$elem_type; $len] {
                // SAFETY: this type is just an overaligned `[T; N]` with
                // potential padding at the end, so pointer casting to a
                // `&[T; N]` is safe.
                //
                // NOTE: This deliberately doesn't just use `&self.0` because it may soon be banned
                // see https://github.com/rust-lang/compiler-team/issues/838
                unsafe { &*(self as *const Self as *const [$elem_type; $len]) }

            }

            /// Returns a mutable array reference containing the entire SIMD vector.
            #[inline]
            $v fn as_mut_array(&mut self) -> &mut [$elem_type; $len] {
                // SAFETY: this type is just an overaligned `[T; N]` with
                // potential padding at the end, so pointer casting to a
                // `&mut [T; N]` is safe.
                //
                // NOTE: This deliberately doesn't just use `&mut self.0` because it may soon be banned
                // see https://github.com/rust-lang/compiler-team/issues/838
                unsafe { &mut *(self as *mut Self as *mut [$elem_type; $len]) }
            }
        }

        $(#[$stability])+
        impl crate::fmt::Debug for $name {
            #[inline]
            fn fmt(&self, f: &mut crate::fmt::Formatter<'_>) -> crate::fmt::Result {
                crate::core_arch::simd::debug_simd_finish(f, stringify!($name), self.0)
            }
        }
    )*);
}

#[allow(unused)]
#[repr(simd)]
pub(crate) struct SimdShuffleIdx<const LEN: usize>(pub(crate) [u32; LEN]);

#[allow(unused)]
macro_rules! simd_shuffle {
    ($x:expr, $y:expr, $idx:expr $(,)?) => {{
        $crate::intrinsics::simd::simd_shuffle(
            $x,
            $y,
            const { $crate::core_arch::macros::SimdShuffleIdx($idx) },
        )
    }};
}

#[allow(unused)]
macro_rules! simd_insert {
    ($x:expr, $idx:expr, $val:expr $(,)?) => {{ $crate::intrinsics::simd::simd_insert($x, const { $idx }, $val) }};
}

#[allow(unused)]
macro_rules! simd_extract {
    ($x:expr, $idx:expr $(,)?) => {{ $crate::intrinsics::simd::simd_extract($x, const { $idx }) }};
    ($x:expr, $idx:expr, $ty:ty $(,)?) => {{ $crate::intrinsics::simd::simd_extract::<_, $ty>($x, const { $idx }) }};
}
