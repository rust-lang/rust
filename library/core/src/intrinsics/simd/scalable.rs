//! Scalable vector compiler intrinsics.
//!
//! In this module, a "vector" is any `#[rustc_scalable_vector]`-annotated type.

/// Numerically casts a vector, elementwise.
///
/// `T` and `U` must be vectors of integers or floats, and must have the same length.
///
/// When casting floats to integers, the result is truncated. Out-of-bounds result lead to UB.
/// When casting integers to floats, the result is rounded.
/// Otherwise, truncates or extends the value, maintaining the sign for signed integers.
///
/// # Safety
/// Casting from integer types is always safe.
/// Casting between two float types is also always safe.
///
/// Casting floats to integers truncates, following the same rules as `to_int_unchecked`.
/// Specifically, each element must:
/// * Not be `NaN`
/// * Not be infinite
/// * Be representable in the return type, after truncating off its fractional part
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sve_cast<T, U>(x: T) -> U;

/// Create a tuple of two vectors.
///
/// `SVecTup` must be a scalable vector tuple (`#[rustc_scalable_vector]`) and `SVec` must be a
/// scalable vector (`#[rustc_scalable_vector(N)]`). `SVecTup` must be a tuple of vectors of
/// type `SVec`.
///
/// Corresponds to Clang's `__builtin_sve_svcreate2*` builtins.
#[rustc_nounwind]
#[rustc_intrinsic]
pub unsafe fn sve_tuple_create2<SVec, SVecTup>(x0: SVec, x1: SVec) -> SVecTup;

/// Create a tuple of three vectors.
///
/// `SVecTup` must be a scalable vector tuple (`#[rustc_scalable_vector]`) and `SVec` must be a
/// scalable vector (`#[rustc_scalable_vector(N)]`). `SVecTup` must be a tuple of vectors of
/// type `SVec`.
///
/// Corresponds to Clang's `__builtin_sve_svcreate3*` builtins.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sve_tuple_create3<SVec, SVecTup>(x0: SVec, x1: SVec, x2: SVec) -> SVecTup;

/// Create a tuple of four vectors.
///
/// `SVecTup` must be a scalable vector tuple (`#[rustc_scalable_vector]`) and `SVec` must be a
/// scalable vector (`#[rustc_scalable_vector(N)]`). `SVecTup` must be a tuple of vectors of
/// type `SVec`.
///
/// Corresponds to Clang's `__builtin_sve_svcreate4*` builtins.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sve_tuple_create4<SVec, SVecTup>(x0: SVec, x1: SVec, x2: SVec, x3: SVec) -> SVecTup;

/// Get one vector from a tuple of vectors.
///
/// `SVecTup` must be a scalable vector tuple (`#[rustc_scalable_vector]`) and `SVec` must be a
/// scalable vector (`#[rustc_scalable_vector(N)]`). `SVecTup` must be a tuple of vectors of
/// type `SVec`.
///
/// Corresponds to Clang's `__builtin_sve_svget*` builtins.
///
/// # Safety
///
/// `IDX` must be in-bounds of the tuple.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sve_tuple_get<SVecTup, SVec, const IDX: i32>(tuple: SVecTup) -> SVec;

/// Change one vector in a tuple of vectors.
///
/// `SVecTup` must be a scalable vector tuple (`#[rustc_scalable_vector]`) and `SVec` must be a
/// scalable vector (`#[rustc_scalable_vector(N)]`). `SVecTup` must be a tuple of vectors of
/// type `SVec`.
///
/// Corresponds to Clang's `__builtin_sve_svset*` builtins.
///
/// # Safety
///
/// `IDX` must be in-bounds of the tuple.
#[rustc_intrinsic]
#[rustc_nounwind]
pub unsafe fn sve_tuple_set<SVecTup, SVec, const IDX: i32>(tuple: SVecTup, x: SVec) -> SVecTup;
