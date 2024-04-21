//! This module reexports the primitive types to allow usage that is not
//! possibly shadowed by other declared types.
//!
//! This is normally only useful in macro generated code.
//!
//! An example of this is when generating a new struct and an impl for it:
//!
//! ```rust,compile_fail
//! pub struct bool;
//!
//! impl QueryId for bool {
//!     const SOME_PROPERTY: bool = true;
//! }
//!
//! # trait QueryId { const SOME_PROPERTY: core::primitive::bool; }
//! ```
//!
//! Note that the `SOME_PROPERTY` associated constant would not compile, as its
//! type `bool` refers to the struct, rather than to the primitive bool type.
//!
//! A correct implementation could look like:
//!
//! ```rust
//! # #[allow(non_camel_case_types)]
//! pub struct bool;
//!
//! impl QueryId for bool {
//!     const SOME_PROPERTY: core::primitive::bool = true;
//! }
//!
//! # trait QueryId { const SOME_PROPERTY: core::primitive::bool; }
//! ```

#[stable(feature = "core_primitive", since = "1.43.0")]
pub use bool;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use char;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use f32;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use f64;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use i128;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use i16;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use i32;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use i64;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use i8;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use isize;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use str;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use u128;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use u16;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use u32;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use u64;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use u8;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use usize;

mod sealed {
    /// This trait being unreachable from outside the crate prevents downstream
    /// implementations, permitting extension of the subtraits without breakage.
    #[unstable(
        feature = "primitive_abstraction_internals",
        reason = "should be replaced by #[sealed] once available",
        issue = "none"
    )]
    pub trait Sealed {}
}

#[macro_use]
mod int_macros; // import int_decl!, int_impl!
#[macro_use]
mod float_macros; // import float_decl!, float_impl!

use crate::{fmt, hash, iter, ops, panic};
use sealed::Sealed;

/// A trait implemented by all *primitive* integer types, signed or unsigned.
///
/// This trait is sealed and cannot be implemented for types outside of the
/// standard library. It is not intended to be a general-purpose numerical
/// abstraction. By being limited to primitive integers, this trait can add
/// new functionality as it is provided without risk of breaking implementors.
#[fundamental]
#[unstable(feature = "primitive_abstraction", issue = "none")]
pub trait Integer:
    'static + sealed::Sealed
    // core::marker
    + Sized + Copy + Clone + Send + Sync + Unpin
    // core::panic
    + panic::UnwindSafe + panic::RefUnwindSafe
    // core::fmt
    + fmt::Debug + fmt::Display + fmt::Binary + fmt::Octal
    + fmt::LowerHex + fmt::UpperHex + fmt::LowerExp + fmt::UpperExp
    // core::default
    + Default
    // core::cmp
    + PartialEq + Eq + PartialOrd + Ord
    // core::hash
    + hash::Hash
    // core::str
    + crate::str::FromStr
    // core::convert
    + TryFrom<u8> + TryInto<u8>
    + TryFrom<u16> + TryInto<u16>
    + TryFrom<u32> + TryInto<u32>
    + TryFrom<u64> + TryInto<u64>
    + TryFrom<u128> + TryInto<u128>
    + TryFrom<usize> + TryInto<usize>
    + TryFrom<i8> + TryInto<i8>
    + TryFrom<i16> + TryInto<i16>
    + TryFrom<i32> + TryInto<i32>
    + TryFrom<i64> + TryInto<i64>
    + TryFrom<i128> + TryInto<i128>
    + TryFrom<isize> + TryInto<isize>
    // core::ops
    + ops::Add<Output = Self> + for<'a> ops::Add<&'a Self, Output = Self>
    + ops::Sub<Output = Self> + for<'a> ops::Sub<&'a Self, Output = Self>
    + ops::Mul<Output = Self> + for<'a> ops::Mul<&'a Self, Output = Self>
    + ops::Div<Output = Self> + for<'a> ops::Div<&'a Self, Output = Self>
    + ops::Rem<Output = Self> + for<'a> ops::Rem<&'a Self, Output = Self>
    + ops::AddAssign + for<'a> ops::AddAssign<&'a Self>
    + ops::SubAssign + for<'a> ops::SubAssign<&'a Self>
    + ops::MulAssign + for<'a> ops::MulAssign<&'a Self>
    + ops::DivAssign + for<'a> ops::DivAssign<&'a Self>
    + ops::RemAssign + for<'a> ops::RemAssign<&'a Self>
    + ops::Not<Output = Self>
    + ops::BitAnd<Output = Self> + for<'a> ops::BitAnd<&'a Self, Output = Self>
    + ops::BitOr<Output = Self> + for<'a> ops::BitOr<&'a Self, Output = Self>
    + ops::BitXor<Output = Self> + for<'a> ops::BitXor<&'a Self, Output = Self>
    + ops::BitAndAssign + for<'a> ops::BitAndAssign<&'a Self>
    + ops::BitOrAssign + for<'a> ops::BitOrAssign<&'a Self>
    + ops::BitXorAssign + for<'a> ops::BitXorAssign<&'a Self>
    + ops::Shl<u8, Output = Self> + for<'a> ops::Shl<&'a u8, Output = Self>
    + ops::Shl<u16, Output = Self> + for<'a> ops::Shl<&'a u16, Output = Self>
    + ops::Shl<u32, Output = Self> + for<'a> ops::Shl<&'a u32, Output = Self>
    + ops::Shl<u64, Output = Self> + for<'a> ops::Shl<&'a u64, Output = Self>
    + ops::Shl<u128, Output = Self> + for<'a> ops::Shl<&'a u128, Output = Self>
    + ops::Shl<usize, Output = Self> + for<'a> ops::Shl<&'a usize, Output = Self>
    + ops::Shl<i8, Output = Self> + for<'a> ops::Shl<&'a i8, Output = Self>
    + ops::Shl<i16, Output = Self> + for<'a> ops::Shl<&'a i16, Output = Self>
    + ops::Shl<i32, Output = Self> + for<'a> ops::Shl<&'a i32, Output = Self>
    + ops::Shl<i64, Output = Self> + for<'a> ops::Shl<&'a i64, Output = Self>
    + ops::Shl<i128, Output = Self> + for<'a> ops::Shl<&'a i128, Output = Self>
    + ops::Shl<isize, Output = Self> + for<'a> ops::Shl<&'a isize, Output = Self>
    + ops::Shl<Self, Output = Self> + for<'a> ops::Shl<&'a Self, Output = Self>
    + ops::Shr<u8, Output = Self> + for<'a> ops::Shr<&'a u8, Output = Self>
    + ops::Shr<u16, Output = Self> + for<'a> ops::Shr<&'a u16, Output = Self>
    + ops::Shr<u32, Output = Self> + for<'a> ops::Shr<&'a u32, Output = Self>
    + ops::Shr<u64, Output = Self> + for<'a> ops::Shr<&'a u64, Output = Self>
    + ops::Shr<u128, Output = Self> + for<'a> ops::Shr<&'a u128, Output = Self>
    + ops::Shr<usize, Output = Self> + for<'a> ops::Shr<&'a usize, Output = Self>
    + ops::Shr<i8, Output = Self> + for<'a> ops::Shr<&'a i8, Output = Self>
    + ops::Shr<i16, Output = Self> + for<'a> ops::Shr<&'a i16, Output = Self>
    + ops::Shr<i32, Output = Self> + for<'a> ops::Shr<&'a i32, Output = Self>
    + ops::Shr<i64, Output = Self> + for<'a> ops::Shr<&'a i64, Output = Self>
    + ops::Shr<i128, Output = Self> + for<'a> ops::Shr<&'a i128, Output = Self>
    + ops::Shr<isize, Output = Self> + for<'a> ops::Shr<&'a isize, Output = Self>
    + ops::Shr<Self, Output = Self> + for<'a> ops::Shr<&'a Self, Output = Self>
    + ops::ShlAssign<u8> + for<'a> ops::ShlAssign<&'a u8>
    + ops::ShlAssign<u16> + for<'a> ops::ShlAssign<&'a u16>
    + ops::ShlAssign<u32> + for<'a> ops::ShlAssign<&'a u32>
    + ops::ShlAssign<u64> + for<'a> ops::ShlAssign<&'a u64>
    + ops::ShlAssign<u128> + for<'a> ops::ShlAssign<&'a u128>
    + ops::ShlAssign<usize> + for<'a> ops::ShlAssign<&'a usize>
    + ops::ShlAssign<i8> + for<'a> ops::ShlAssign<&'a i8>
    + ops::ShlAssign<i16> + for<'a> ops::ShlAssign<&'a i16>
    + ops::ShlAssign<i32> + for<'a> ops::ShlAssign<&'a i32>
    + ops::ShlAssign<i64> + for<'a> ops::ShlAssign<&'a i64>
    + ops::ShlAssign<i128> + for<'a> ops::ShlAssign<&'a i128>
    + ops::ShlAssign<isize> + for<'a> ops::ShlAssign<&'a isize>
    + ops::ShlAssign<Self> + for<'a> ops::ShlAssign<&'a Self>
    + ops::ShrAssign<u8> + for<'a> ops::ShrAssign<&'a u8>
    + ops::ShrAssign<u16> + for<'a> ops::ShrAssign<&'a u16>
    + ops::ShrAssign<u32> + for<'a> ops::ShrAssign<&'a u32>
    + ops::ShrAssign<u64> + for<'a> ops::ShrAssign<&'a u64>
    + ops::ShrAssign<u128> + for<'a> ops::ShrAssign<&'a u128>
    + ops::ShrAssign<usize> + for<'a> ops::ShrAssign<&'a usize>
    + ops::ShrAssign<i8> + for<'a> ops::ShrAssign<&'a i8>
    + ops::ShrAssign<i16> + for<'a> ops::ShrAssign<&'a i16>
    + ops::ShrAssign<i32> + for<'a> ops::ShrAssign<&'a i32>
    + ops::ShrAssign<i64> + for<'a> ops::ShrAssign<&'a i64>
    + ops::ShrAssign<i128> + for<'a> ops::ShrAssign<&'a i128>
    + ops::ShrAssign<isize> + for<'a> ops::ShrAssign<&'a isize>
    + ops::ShrAssign<Self> + for<'a> ops::ShrAssign<&'a Self>
    // core::iter
    + iter::Sum<Self> + for<'a> iter::Sum<&'a Self>
    + iter::Product<Self> + for<'a> iter::Product<&'a Self>
{
    int_decl!();
}

/// A trait implemented by all *primitive* floating point types.
///
/// This trait is sealed and cannot be implemented for types outside of the
/// standard library. It is not intended to be a general-purpose numerical
/// abstraction. By being limited to primitive floating point, this trait can add
/// new functionality as it is provided without risk of breaking implementors.
#[fundamental]
#[unstable(feature = "primitive_abstraction", issue = "none")]
pub trait Float:
    'static + sealed::Sealed
    // core::marker
    + Sized + Copy + Clone + Send + Sync + Unpin
    // core::panic
    + panic::UnwindSafe + panic::RefUnwindSafe
    // core::fmt
    + fmt::Debug // + fmt::Display + fmt::LowerExp + fmt::UpperExp
    // core::default
    + Default
    // core::cmp
    + PartialEq + PartialOrd
    // core::ops
    + ops::Add<Output = Self> + for<'a> ops::Add<&'a Self, Output = Self>
    + ops::Sub<Output = Self> + for<'a> ops::Sub<&'a Self, Output = Self>
    + ops::Mul<Output = Self> + for<'a> ops::Mul<&'a Self, Output = Self>
    + ops::Div<Output = Self> + for<'a> ops::Div<&'a Self, Output = Self>
    + ops::Rem<Output = Self> + for<'a> ops::Rem<&'a Self, Output = Self>
    + ops::AddAssign + for<'a> ops::AddAssign<&'a Self>
    + ops::SubAssign + for<'a> ops::SubAssign<&'a Self>
    + ops::MulAssign + for<'a> ops::MulAssign<&'a Self>
    + ops::DivAssign + for<'a> ops::DivAssign<&'a Self>
    + ops::RemAssign + for<'a> ops::RemAssign<&'a Self>
    + ops::Neg
{
    float_decl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for u8 {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Integer for u8 {
    int_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for u16 {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Integer for u16 {
    int_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for u32 {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Integer for u32 {
    int_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for u64 {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Integer for u64 {
    int_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for u128 {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Integer for u128 {
    int_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for usize {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Integer for usize {
    int_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for i8 {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Integer for i8 {
    int_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for i16 {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Integer for i16 {
    int_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for i32 {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Integer for i32 {
    int_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for i64 {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Integer for i64 {
    int_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for i128 {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Integer for i128 {
    int_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for isize {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Integer for isize {
    int_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for f16 {}

#[cfg(not(bootstrap))]
#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Float for f16 {
    float_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for f32 {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Float for f32 {
    float_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for f64 {}

#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Float for f64 {
    float_impl!();
}

#[unstable(feature = "primitive_abstraction_internals", issue = "none")]
impl Sealed for f128 {}

#[cfg(not(bootstrap))]
#[unstable(feature = "primitive_abstraction", issue = "none")]
impl Float for f128 {
    float_impl!();
}
