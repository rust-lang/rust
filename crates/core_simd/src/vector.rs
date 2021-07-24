#[macro_use]
mod vector_impl;

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

    /// Generates a SIMD vector with the same value in every lane.
    #[must_use]
    fn splat(val: Self::Scalar) -> Self;
}
