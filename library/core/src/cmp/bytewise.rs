use crate::num::{NonZeroI128, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI8, NonZeroIsize};
use crate::num::{NonZeroU128, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU8, NonZeroUsize};

/// Types where `==` & `!=` are equivalent to comparing their underlying bytes.
///
/// Importantly, this means no floating-point types, as those have different
/// byte representations (like `-0` and `+0`) which compare as the same.
/// Since byte arrays are `Eq`, that implies that these types are probably also
/// `Eq`, but that's not technically required to use this trait.
///
/// `Rhs` is *de facto* always `Self`, but the separate parameter is important
/// to avoid the `specializing impl repeats parameter` error when consuming this.
///
/// # Safety
///
/// - `Self` and `Rhs` have no padding.
/// - `Self` and `Rhs` have the same layout (size and alignment).
/// - Neither `Self` nor `Rhs` have provenance, so integer comparisons are correct.
/// - `<Self as PartialEq<Rhs>>::{eq,ne}` are equivalent to comparing the bytes.
#[rustc_specialization_trait]
pub(crate) unsafe trait BytewiseEq<Rhs = Self>: PartialEq<Rhs> + Sized {}

macro_rules! is_bytewise_comparable {
    ($($t:ty),+ $(,)?) => {$(
        unsafe impl BytewiseEq for $t {}
    )+};
}

// SAFETY: All the ordinary integer types have no padding, and are not pointers.
is_bytewise_comparable!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

// SAFETY: These have *niches*, but no *padding* and no *provenance*,
// so we can compare them directly.
is_bytewise_comparable!(bool, char, super::Ordering);

// SAFETY: Similarly, the non-zero types have a niche, but no undef and no pointers,
// and they compare like their underlying numeric type.
is_bytewise_comparable!(
    NonZeroU8,
    NonZeroU16,
    NonZeroU32,
    NonZeroU64,
    NonZeroU128,
    NonZeroUsize,
    NonZeroI8,
    NonZeroI16,
    NonZeroI32,
    NonZeroI64,
    NonZeroI128,
    NonZeroIsize,
);

// SAFETY: The NonZero types have the "null" optimization guaranteed, and thus
// are also safe to equality-compare bitwise inside an `Option`.
// The way `PartialOrd` is defined for `Option` means that this wouldn't work
// for `<` or `>` on the signed types, but since we only do `==` it's fine.
is_bytewise_comparable!(
    Option<NonZeroU8>,
    Option<NonZeroU16>,
    Option<NonZeroU32>,
    Option<NonZeroU64>,
    Option<NonZeroU128>,
    Option<NonZeroUsize>,
    Option<NonZeroI8>,
    Option<NonZeroI16>,
    Option<NonZeroI32>,
    Option<NonZeroI64>,
    Option<NonZeroI128>,
    Option<NonZeroIsize>,
);

macro_rules! is_bytewise_comparable_array_length {
    ($($n:literal),+ $(,)?) => {$(
        // SAFETY: Arrays have no padding between elements, so if the elements are
        // `BytewiseEq`, then the whole array can be too.
        unsafe impl<T: BytewiseEq<U>, U> BytewiseEq<[U; $n]> for [T; $n] {}
    )+};
}

// Frustratingly, this can't be made const-generic as it gets
//    error: specializing impl repeats parameter `N`
// so just do it for a couple of plausibly-common ones.
is_bytewise_comparable_array_length!(0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64);
