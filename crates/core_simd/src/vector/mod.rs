mod float;
mod int;
mod uint;

pub use float::*;
pub use int::*;
pub use uint::*;

// Vectors of pointers are not for public use at the current time.
pub(crate) mod ptr;

mod sealed {
    pub trait Sealed {}
}

/// A representation of a vector as an "array" with indices, implementing
/// operations applicable to any vector type based solely on "having lanes",
/// and describing relationships between vector and scalar types.
pub trait Vector: sealed::Sealed {
    /// The scalar type in every lane of this vector type.
    type Scalar: Copy + Sized;

    /// The number of lanes for this vector.
    const LANES: usize;

    // Implementation detail until the compiler can support bitmasks of any integer width
    #[doc(hidden)]
    type BitMask: Into<u64>;

    /// Generates a SIMD vector with the same value in every lane.
    #[must_use]
    fn splat(val: Self::Scalar) -> Self;

    /// Returns a slice containing the entire SIMD vector.
    fn as_slice(&self) -> &[Self::Scalar];

    /// Returns a mutable slice containing the entire SIMD vector.
    fn as_mut_slice(&mut self) -> &mut [Self::Scalar];
}

macro_rules! impl_vector_for {
    ($simd:ident {type Scalar = $scalar:ident;}) => {
        impl_vector_for! { $simd<1> { type Scalar = $scalar; type BitMask = u8; } }
        impl_vector_for! { $simd<2> { type Scalar = $scalar; type BitMask = u8; } }
        impl_vector_for! { $simd<4> { type Scalar = $scalar; type BitMask = u8; } }
        impl_vector_for! { $simd<8> { type Scalar = $scalar; type BitMask = u8; } }
        impl_vector_for! { $simd<16> { type Scalar = $scalar; type BitMask = u16; } }
        impl_vector_for! { $simd<32> { type Scalar = $scalar; type BitMask = u32; } }
    };
    ($simd:ident<$lanes:literal> {type Scalar = $scalar:ident; type BitMask = $bitmask:ident; }) => {
        impl sealed::Sealed for $simd<$lanes> {}

        impl Vector for $simd<$lanes> {
            type Scalar = $scalar;
            const LANES: usize = $lanes;

            type BitMask = $bitmask;

            #[inline]
            fn splat(val: Self::Scalar) -> Self {
                Self::splat(val)
            }

            #[inline]
            fn as_slice(&self) -> &[Self::Scalar] {
                self.as_slice()
            }

            #[inline]
            fn as_mut_slice(&mut self) -> &mut [Self::Scalar] {
                self.as_mut_slice()
            }
        }
    };
}

impl_vector_for! {
    SimdUsize {
        type Scalar = usize;
    }
}

impl_vector_for! {
    SimdIsize {
        type Scalar = isize;
    }
}

impl_vector_for! {
    SimdI8 {
        type Scalar = i8;
    }
}

impl_vector_for! {
    SimdI16 {
        type Scalar = i16;
    }
}

impl_vector_for! {
    SimdI32 {
        type Scalar = i32;
    }
}

impl_vector_for! {
    SimdI64 {
        type Scalar = i64;
    }
}

impl_vector_for! {
    SimdU8 {
        type Scalar = u8;
    }
}

impl_vector_for! {
    SimdU16 {
        type Scalar = u16;
    }
}

impl_vector_for! {
    SimdU32 {
        type Scalar = u32;
    }
}

impl_vector_for! {
    SimdU64 {
        type Scalar = u64;
    }
}

impl_vector_for! {
    SimdF32 {
        type Scalar = f32;
    }
}

impl_vector_for! {
    SimdF64 {
        type Scalar = f64;
    }
}
