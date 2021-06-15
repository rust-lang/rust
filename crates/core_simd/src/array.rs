use crate::masks::*;
use crate::vector::*;

/// A representation of a vector as an "array" with indices, implementing
/// operations applicable to any vector type based solely on "having lanes",
/// and describing relationships between vector and scalar types.
pub trait SimdArray<const LANES: usize>: crate::LanesAtMost32
where
    SimdUsize<LANES>: crate::LanesAtMost32,
    SimdIsize<LANES>: crate::LanesAtMost32,
    MaskSize<LANES>: crate::Mask,
    Self: Sized,
{
    /// The scalar type in every lane of this vector type.
    type Scalar: Copy + Sized;

    /// Generates a SIMD vector with the same value in every lane.
    #[must_use]
    fn splat(val: Self::Scalar) -> Self;
}

macro_rules! impl_simdarray_for {
    ($simd:ident {type Scalar = $scalar:ident;}) => {
        impl<const LANES: usize> SimdArray<LANES> for $simd<LANES>
            where SimdUsize<LANES>: crate::LanesAtMost32,
            SimdIsize<LANES>: crate::LanesAtMost32,
            MaskSize<LANES>: crate::Mask,
            Self: crate::LanesAtMost32,
        {
            type Scalar = $scalar;

            #[must_use]
            #[inline]
            fn splat(val: Self::Scalar) -> Self {
                [val; LANES].into()
            }
        }
    };

    ($simd:ident $impl:tt) => {
        impl<const LANES: usize> SimdArray<LANES> for $simd<LANES>
            where SimdUsize<LANES>: crate::LanesAtMost32,
            SimdIsize<LANES>: crate::LanesAtMost32,
            MaskSize<LANES>: crate::Mask,
            Self: crate::LanesAtMost32,
        $impl
    }
}

impl_simdarray_for! {
    SimdUsize {
        type Scalar = usize;
    }
}

impl_simdarray_for! {
    SimdIsize {
        type Scalar = isize;
    }
}

impl_simdarray_for! {
    SimdI8 {
        type Scalar = i8;
    }
}

impl_simdarray_for! {
    SimdI16 {
        type Scalar = i16;
    }
}

impl_simdarray_for! {
    SimdI32 {
        type Scalar = i32;
    }
}

impl_simdarray_for! {
    SimdI64 {
        type Scalar = i64;
    }
}

impl_simdarray_for! {
    SimdU8 {
        type Scalar = u8;
    }
}

impl_simdarray_for! {
    SimdU16 {
        type Scalar = u16;
    }
}

impl_simdarray_for! {
    SimdU32 {
        type Scalar = u32;
    }
}

impl_simdarray_for! {
    SimdU64 {
        type Scalar = u64;
    }
}

impl_simdarray_for! {
    SimdF32 {
        type Scalar = f32;
    }
}

impl_simdarray_for! {
    SimdF64 {
        type Scalar = f64;
    }
}
