use crate::convert::TryInto;

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<[B; N]> for [A; N]
where
    A: PartialEq<B>,
{
    #[inline]
    fn eq(&self, other: &[B; N]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &[B; N]) -> bool {
        self[..] != other[..]
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
