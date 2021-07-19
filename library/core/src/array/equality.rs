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
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &[B]) -> bool {
        self[..] != other[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<[A; N]> for [B]
where
    B: PartialEq<A>,
{
    #[inline]
    fn eq(&self, other: &[A; N]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &[A; N]) -> bool {
        self[..] != other[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<&[B]> for [A; N]
where
    A: PartialEq<B>,
{
    #[inline]
    fn eq(&self, other: &&[B]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &&[B]) -> bool {
        self[..] != other[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<[A; N]> for &[B]
where
    B: PartialEq<A>,
{
    #[inline]
    fn eq(&self, other: &[A; N]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &[A; N]) -> bool {
        self[..] != other[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<&mut [B]> for [A; N]
where
    A: PartialEq<B>,
{
    #[inline]
    fn eq(&self, other: &&mut [B]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &&mut [B]) -> bool {
        self[..] != other[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<[A; N]> for &mut [B]
where
    B: PartialEq<A>,
{
    #[inline]
    fn eq(&self, other: &[A; N]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &[A; N]) -> bool {
        self[..] != other[..]
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

impl<T: PartialEq<U> + IsRawEqComparable<U>, U, const N: usize> SpecArrayEq<U, N> for T {
    #[cfg(bootstrap)]
    fn spec_eq(a: &[T; N], b: &[U; N]) -> bool {
        a[..] == b[..]
    }
    #[cfg(not(bootstrap))]
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
unsafe trait IsRawEqComparable<U> {}

macro_rules! is_raw_comparable {
    ($($t:ty),+) => {$(
        unsafe impl IsRawEqComparable<$t> for $t {}
    )+};
}
is_raw_comparable!(bool, char, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
