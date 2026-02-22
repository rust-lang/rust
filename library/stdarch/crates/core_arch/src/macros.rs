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
            /// Put the same value in every lane.
            #[inline(always)]
            $v fn splat(value: $elem_type) -> $name {
                unsafe { $crate::intrinsics::simd::simd_splat(value) }
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
                crate::core_arch::simd::debug_simd_finish(f, stringify!($name), self.as_array())
            }
        }

        $(#[$stability])+
        impl crate::convert::From<crate::core_arch::simd::Simd<$elem_type, $len>> for $name {
            #[inline(always)]
            fn from(simd: crate::core_arch::simd::Simd<$elem_type, $len>) -> Self {
                unsafe { crate::mem::transmute(simd) }
            }
        }

        $(#[$stability])+
        impl crate::convert::From<$name> for crate::core_arch::simd::Simd<$elem_type, $len> {
            #[inline(always)]
            fn from(simd: $name) -> Self {
                unsafe { crate::mem::transmute(simd) }
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

#[allow(unused)]
macro_rules! simd_masked_load {
    ($align:expr, $mask:expr, $ptr:expr, $default:expr) => {
        $crate::intrinsics::simd::simd_masked_load::<_, _, _, { $align }>($mask, $ptr, $default)
    };
}

#[allow(unused)]
macro_rules! simd_masked_store {
    ($align:expr, $mask:expr, $ptr:expr, $default:expr) => {
        $crate::intrinsics::simd::simd_masked_store::<_, _, _, { $align }>($mask, $ptr, $default)
    };
}

/// The first N even indices `[0, 2, 4, ...]`.
pub(crate) const fn even<const N: usize>() -> [u32; N] {
    let mut out = [0u32; N];
    let mut i = 0usize;
    while i < N {
        out[i] = (2 * i) as u32;
        i += 1;
    }
    out
}

/// The first N odd indices `[1, 3, 5, ...]`.
pub(crate) const fn odd<const N: usize>() -> [u32; N] {
    let mut out = [0u32; N];
    let mut i = 0usize;
    while i < N {
        out[i] = (2 * i + 1) as u32;
        i += 1;
    }
    out
}

/// Multiples of N offset by K `[K, K+N, K+2N, ...]`.
pub(crate) const fn deinterleave_mask<const LANES: usize, const N: usize, const K: usize>()
-> [u32; LANES] {
    let mut out = [0u32; LANES];
    let mut i = 0usize;
    while i < LANES {
        out[i] = (i * N + K) as u32;
        i += 1;
    }
    out
}

#[allow(unused)]
macro_rules! deinterleaving_load {
    ($elem:ty, $lanes:literal, 2, $ptr:expr) => {{
        use $crate::core_arch::macros::deinterleave_mask;
        use $crate::core_arch::simd::Simd;
        use $crate::{mem::transmute, ptr};

        type V = Simd<$elem, $lanes>;
        type W = Simd<$elem, { $lanes * 2 }>;

        let w: W = ptr::read_unaligned($ptr as *const W);

        let v0: V = simd_shuffle!(w, w, deinterleave_mask::<$lanes, 2, 0>());
        let v1: V = simd_shuffle!(w, w, deinterleave_mask::<$lanes, 2, 1>());

        transmute((v0, v1))
    }};

    ($elem:ty, $lanes:literal, 3, $ptr:expr) => {{
        use $crate::core_arch::macros::deinterleave_mask;
        use $crate::core_arch::simd::Simd;
        use $crate::{mem::transmute, ptr};

        type V = Simd<$elem, $lanes>;
        type W = Simd<$elem, { $lanes * 3 }>;

        let w: W = ptr::read_unaligned($ptr as *const W);

        let v0: V = simd_shuffle!(w, w, deinterleave_mask::<$lanes, 3, 0>());
        let v1: V = simd_shuffle!(w, w, deinterleave_mask::<$lanes, 3, 1>());
        let v2: V = simd_shuffle!(w, w, deinterleave_mask::<$lanes, 3, 2>());

        transmute((v0, v1, v2))
    }};

    ($elem:ty, $lanes:literal, 4, $ptr:expr) => {{
        use $crate::core_arch::macros::deinterleave_mask;
        use $crate::core_arch::simd::Simd;
        use $crate::{mem::transmute, ptr};

        type V = Simd<$elem, $lanes>;
        type W = Simd<$elem, { $lanes * 4 }>;

        let w: W = ptr::read_unaligned($ptr as *const W);

        let v0: V = simd_shuffle!(w, w, deinterleave_mask::<$lanes, 4, 0>());
        let v1: V = simd_shuffle!(w, w, deinterleave_mask::<$lanes, 4, 1>());
        let v2: V = simd_shuffle!(w, w, deinterleave_mask::<$lanes, 4, 2>());
        let v3: V = simd_shuffle!(w, w, deinterleave_mask::<$lanes, 4, 3>());

        transmute((v0, v1, v2, v3))
    }};
}

#[allow(unused)]
pub(crate) use deinterleaving_load;
