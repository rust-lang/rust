use crate::cmp::BytewiseEq;
use crate::convert::TryInto;

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

impl<T: BytewiseEq<U>, U, const N: usize> SpecArrayEq<U, N> for T {
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
