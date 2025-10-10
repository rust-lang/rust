use crate::cmp::BytewiseEq;

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T, U, const N: usize> const PartialEq<[U; N]> for [T; N]
where
    T: [const] PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        SpecArrayEq::spec_eq(self, other)
    }
    #[inline]
    fn ne(&self, other: &[U; N]) -> bool {
        SpecArrayEq::spec_ne(self, other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T, U, const N: usize> const PartialEq<[U]> for [T; N]
where
    T: [const] PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U]) -> bool {
        match other.as_array::<N>() {
            Some(b) => *self == *b,
            None => false,
        }
    }
    #[inline]
    fn ne(&self, other: &[U]) -> bool {
        match other.as_array::<N>() {
            Some(b) => *self != *b,
            None => true,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T, U, const N: usize> const PartialEq<[U; N]> for [T]
where
    T: [const] PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        match self.as_array::<N>() {
            Some(b) => *b == *other,
            None => false,
        }
    }
    #[inline]
    fn ne(&self, other: &[U; N]) -> bool {
        match self.as_array::<N>() {
            Some(b) => *b != *other,
            None => true,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T, U, const N: usize> const PartialEq<&[U]> for [T; N]
where
    T: [const] PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&[U]) -> bool {
        *self == **other
    }
    #[inline]
    fn ne(&self, other: &&[U]) -> bool {
        *self != **other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T, U, const N: usize> const PartialEq<[U; N]> for &[T]
where
    T: [const] PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        **self == *other
    }
    #[inline]
    fn ne(&self, other: &[U; N]) -> bool {
        **self != *other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T, U, const N: usize> const PartialEq<&mut [U]> for [T; N]
where
    T: [const] PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&mut [U]) -> bool {
        *self == **other
    }
    #[inline]
    fn ne(&self, other: &&mut [U]) -> bool {
        *self != **other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T, U, const N: usize> const PartialEq<[U; N]> for &mut [T]
where
    T: [const] PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        **self == *other
    }
    #[inline]
    fn ne(&self, other: &[U; N]) -> bool {
        **self != *other
    }
}

// NOTE: some less important impls are omitted to reduce code bloat
// __impl_slice_eq2! { [A; $N], &'b [B; $N] }
// __impl_slice_eq2! { [A; $N], &'b mut [B; $N] }

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T: [const] Eq, const N: usize> const Eq for [T; N] {}

#[const_trait]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
trait SpecArrayEq<Other, const N: usize>: Sized {
    fn spec_eq(a: &[Self; N], b: &[Other; N]) -> bool;
    fn spec_ne(a: &[Self; N], b: &[Other; N]) -> bool;
}

#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T: [const] PartialEq<Other>, Other, const N: usize> const SpecArrayEq<Other, N> for T {
    default fn spec_eq(a: &[Self; N], b: &[Other; N]) -> bool {
        a[..] == b[..]
    }
    default fn spec_ne(a: &[Self; N], b: &[Other; N]) -> bool {
        a[..] != b[..]
    }
}

#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T: [const] BytewiseEq<U>, U, const N: usize> const SpecArrayEq<U, N> for T {
    fn spec_eq(a: &[T; N], b: &[U; N]) -> bool {
        // SAFETY: Arrays are compared element-wise, and don't add any padding
        // between elements, so when the elements are `BytewiseEq`, we can
        // compare the entire array at once.
        unsafe { crate::intrinsics::raw_eq(a, crate::mem::transmute(b)) }
    }
    fn spec_ne(a: &[T; N], b: &[U; N]) -> bool {
        !Self::spec_eq(a, b)
    }
}
