use crate::convert::TryInto;
use crate::num::{NonZeroI128, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI8, NonZeroIsize};
use crate::num::{NonZeroU128, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU8, NonZeroUsize};

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<[B; N]> for [A; N]
where
    A: PartialEq<B>,
{
    #[inline]
    fn eq(&self, other: &[B; N]) -> bool {
        SpecArrayEq::spec_eq(self, other)
    }
    #[inline]
    fn ne(&self, other: &[B; N]) -> bool {
        SpecArrayEq::spec_ne(self, other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<[B]> for [A; N]
where
    A: PartialEq<B>,
{
    #[inline]
    fn eq(&self, other: &[B]) -> bool {
        let b: Result<&[B; N], _> = other.try_into();
        match b {
            Ok(b) => *self == *b,
            Err(_) => false,
        }
    }
    #[inline]
    fn ne(&self, other: &[B]) -> bool {
        let b: Result<&[B; N], _> = other.try_into();
        match b {
            Ok(b) => *self != *b,
            Err(_) => true,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<[A; N]> for [B]
where
    B: PartialEq<A>,
{
    #[inline]
    fn eq(&self, other: &[A; N]) -> bool {
        let b: Result<&[B; N], _> = self.try_into();
        match b {
            Ok(b) => *b == *other,
            Err(_) => false,
        }
    }
    #[inline]
    fn ne(&self, other: &[A; N]) -> bool {
        let b: Result<&[B; N], _> = self.try_into();
        match b {
            Ok(b) => *b != *other,
            Err(_) => true,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<&[B]> for [A; N]
where
    A: PartialEq<B>,
{
    #[inline]
    fn eq(&self, other: &&[B]) -> bool {
        *self == **other
    }
    #[inline]
    fn ne(&self, other: &&[B]) -> bool {
        *self != **other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<[A; N]> for &[B]
where
    B: PartialEq<A>,
{
    #[inline]
    fn eq(&self, other: &[A; N]) -> bool {
        **self == *other
    }
    #[inline]
    fn ne(&self, other: &[A; N]) -> bool {
        **self != *other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<&mut [B]> for [A; N]
where
    A: PartialEq<B>,
{
    #[inline]
    fn eq(&self, other: &&mut [B]) -> bool {
        *self == **other
    }
    #[inline]
    fn ne(&self, other: &&mut [B]) -> bool {
        *self != **other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<[A; N]> for &mut [B]
where
    B: PartialEq<A>,
{
    #[inline]
    fn eq(&self, other: &[A; N]) -> bool {
        **self == *other
    }
    #[inline]
    fn ne(&self, other: &[A; N]) -> bool {
        **self != *other
    }
}

// NOTE: some less important impls are omitted to reduce code bloat
// __impl_slice_eq2! { [A; $N], &'b [B; $N] }
// __impl_slice_eq2! { [A; $N], &'b mut [B; $N] }

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq, const N: usize> Eq for [T; N] {}

trait SpecArrayEq<Other, const N: usize>: Sized {
    fn spec_eq(a: &[Self; N], b: &[Other; N]) -> bool;
    fn spec_ne(a: &[Self; N], b: &[Other; N]) -> bool;
}

impl<T: PartialEq<Other>, Other, const N: usize> SpecArrayEq<Other, N> for T {
    default fn spec_eq(a: &[Self; N], b: &[Other; N]) -> bool {
        a[..] == b[..]
    }
    default fn spec_ne(a: &[Self; N], b: &[Other; N]) -> bool {
        a[..] != b[..]
    }
}

impl<T: IsRawEqComparable<U>, U, const N: usize> SpecArrayEq<U, N> for T {
    fn spec_eq(a: &[T; N], b: &[U; N]) -> bool {
        // SAFETY: This is why `IsRawEqComparable` is an `unsafe trait`.
        unsafe {
            let b = &*b.as_ptr().cast::<[T; N]>();
            crate::intrinsics::raw_eq(a, b)
        }
    }
    fn spec_ne(a: &[T; N], b: &[U; N]) -> bool {
        !Self::spec_eq(a, b)
    }
}

/// `U` exists on here mostly because `min_specialization` didn't let me
/// repeat the `T` type parameter in the above specialization, so instead
/// the `T == U` constraint comes from the impls on this.
/// # Safety
/// - Neither `Self` nor `U` has any padding.
/// - `Self` and `U` have the same layout.
/// - `Self: PartialEq<U>` is byte-wise (this means no floats, among other things)
#[rustc_specialization_trait]
unsafe trait IsRawEqComparable<U>: PartialEq<U> {}

macro_rules! is_raw_eq_comparable {
    ($($t:ty),+ $(,)?) => {$(
        unsafe impl IsRawEqComparable<$t> for $t {}
    )+};
}

// SAFETY: All the ordinary integer types allow all bit patterns as distinct values
is_raw_eq_comparable!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

// SAFETY: bool and char have *niches*, but no *padding*, so this is sound
is_raw_eq_comparable!(bool, char);

// SAFETY: Similarly, the non-zero types have a niche, but no undef,
// and they compare like their underlying numeric type.
is_raw_eq_comparable!(
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
is_raw_eq_comparable!(
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
