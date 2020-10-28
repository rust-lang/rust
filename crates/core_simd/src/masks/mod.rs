//! Types and traits associated with masking lanes of vectors.

pub mod wide;

mod ops;
pub use ops::*;

pub(crate) trait MaskImpl {
    type Mask;
}

impl MaskImpl for [u8; 8] {
    type Mask = wide::m8x8;
}

impl MaskImpl for [u8; 16] {
    type Mask = wide::m8x16;
}

impl MaskImpl for [u8; 32] {
    type Mask = wide::m8x32;
}

impl MaskImpl for [u8; 64] {
    type Mask = wide::m8x64;
}

impl MaskImpl for [u16; 4] {
    type Mask = wide::m16x4;
}

impl MaskImpl for [u16; 8] {
    type Mask = wide::m16x8;
}

impl MaskImpl for [u16; 16] {
    type Mask = wide::m16x16;
}

impl MaskImpl for [u16; 32] {
    type Mask = wide::m16x32;
}

impl MaskImpl for [u32; 2] {
    type Mask = wide::m32x2;
}

impl MaskImpl for [u32; 4] {
    type Mask = wide::m32x4;
}

impl MaskImpl for [u32; 8] {
    type Mask = wide::m32x8;
}

impl MaskImpl for [u32; 16] {
    type Mask = wide::m32x16;
}

impl MaskImpl for [u64; 2] {
    type Mask = wide::m64x2;
}

impl MaskImpl for [u64; 4] {
    type Mask = wide::m64x4;
}

impl MaskImpl for [u64; 8] {
    type Mask = wide::m64x8;
}

impl MaskImpl for [u128; 2] {
    type Mask = wide::m128x2;
}

impl MaskImpl for [u128; 4] {
    type Mask = wide::m128x4;
}

impl MaskImpl for [usize; 2] {
    type Mask = wide::msizex2;
}

impl MaskImpl for [usize; 4] {
    type Mask = wide::msizex4;
}

impl MaskImpl for [usize; 8] {
    type Mask = wide::msizex8;
}

macro_rules! define_opaque_mask {
    {
        $(#[$attr:meta])*
        struct $name:ident([$width:ty; $lanes:tt]);
    } => {
        $(#[$attr])*
        #[allow(non_camel_case_types)]
        pub struct $name(<[$width; $lanes] as crate::masks::MaskImpl>::Mask);

        impl $name {
            pub(crate) fn new_from_inner(inner: <[$width; $lanes] as crate::masks::MaskImpl>::Mask) -> Self {
                Self(inner)
            }

            /// Construct a mask by setting all lanes to the given value.
            pub fn splat(value: bool) -> Self {
                Self(<[$width; $lanes] as crate::masks::MaskImpl>::Mask::splat(value.into()))
            }

            call_counting_args! { $lanes => define_opaque_mask => new [$width; $lanes] }

            /// Tests the value of the specified lane.
            ///
            /// # Panics
            /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
            #[inline]
            pub fn test(&self, lane: usize) -> bool {
                self.0.test(lane)
            }

            /// Sets the value of the specified lane.
            ///
            /// # Panics
            /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
            #[inline]
            pub fn set(&mut self, lane: usize, value: bool) {
                self.0.set(lane, value);
            }
        }

        impl Copy for $name {}

        impl Clone for $name {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl Default for $name {
            #[inline]
            fn default() -> Self {
                Self::splat(false)
            }
        }

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl PartialOrd for $name {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }
    };
    { new [$width:ty; $lanes:tt] $($var:ident)* } => {
        /// Construct a vector by setting each lane to the given values.
        #[allow(clippy::too_many_arguments)]
        #[inline]
        pub const fn new($($var: bool),*) -> Self {
            Self(<[$width; $lanes] as crate::masks::MaskImpl>::Mask::new_from_bool($($var),*))
        }
    }
}

pub(crate) mod opaque {
    define_opaque_mask! {
        /// Mask for 8 8-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask8x8([u8; 8]);
    }

    define_opaque_mask! {
        /// Mask for 16 8-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask8x16([u8; 16]);
    }

    define_opaque_mask! {
        /// Mask for 32 8-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask8x32([u8; 32]);
    }

    define_opaque_mask! {
        /// Mask for 64 8-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask8x64([u8; 64]);
    }

    define_opaque_mask! {
        /// Mask for 4 16-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask16x4([u16; 4]);
    }

    define_opaque_mask! {
        /// Mask for 8 16-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask16x8([u16; 8]);
    }

    define_opaque_mask! {
        /// Mask for 16 16-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask16x16([u16; 16]);
    }

    define_opaque_mask! {
        /// Mask for 32 16-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask16x32([u16; 32]);
    }

    define_opaque_mask! {
        /// Mask for 2 32-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask32x2([u32; 2]);
    }

    define_opaque_mask! {
        /// Mask for 4 32-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask32x4([u32; 4]);
    }

    define_opaque_mask! {
        /// Mask for 8 32-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask32x8([u32; 8]);
    }

    define_opaque_mask! {
        /// Mask for 16 32-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask32x16([u32; 16]);
    }

    define_opaque_mask! {
        /// Mask for 2 64-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask64x2([u64; 2]);
    }

    define_opaque_mask! {
        /// Mask for 4 64-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask64x4([u64; 4]);
    }

    define_opaque_mask! {
        /// Mask for 8 64-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask64x8([u64; 8]);
    }

    define_opaque_mask! {
        /// Mask for 2 128-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask128x2([u128; 2]);
    }

    define_opaque_mask! {
        /// Mask for 4 128-bit lanes.
        ///
        /// The layout of this type is unspecified.
        struct mask128x4([u128; 4]);
    }

    define_opaque_mask! {
        /// Mask for 2 `isize`-wide lanes.
        ///
        /// The layout of this type is unspecified.
        struct masksizex2([usize; 2]);
    }

    define_opaque_mask! {
        /// Mask for 4 `isize`-wide lanes.
        ///
        /// The layout of this type is unspecified.
        struct masksizex4([usize; 4]);
    }

    define_opaque_mask! {
        /// Mask for 8 `isize`-wide lanes.
        ///
        /// The layout of this type is unspecified.
        struct masksizex8([usize; 8]);
    }
}
