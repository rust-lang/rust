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
