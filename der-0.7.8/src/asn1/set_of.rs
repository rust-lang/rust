//! ASN.1 `SET OF` support.
//!
//! # Ordering Notes
//!
//! Some DER serializer implementations fail to properly sort elements of a
//! `SET OF`. This is technically non-canonical, but occurs frequently
//! enough that most DER decoders tolerate it. Unfortunately because
//! of that, we must also follow suit.
//!
//! However, all types in this module sort elements of a set at decode-time,
//! ensuring they'll be in the proper order if reserialized.

use crate::{
    arrayvec, ord::iter_cmp, ArrayVec, Decode, DecodeValue, DerOrd, Encode, EncodeValue, Error,
    ErrorKind, FixedTag, Header, Length, Reader, Result, Tag, ValueOrd, Writer,
};
use core::cmp::Ordering;

#[cfg(feature = "alloc")]
use {alloc::vec::Vec, core::slice};

/// ASN.1 `SET OF` backed by an array.
///
/// This type implements an append-only `SET OF` type which is stack-based
/// and does not depend on `alloc` support.
// TODO(tarcieri): use `ArrayVec` when/if it's merged into `core`
// See: https://github.com/rust-lang/rfcs/pull/2990
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct SetOf<T, const N: usize>
where
    T: DerOrd,
{
    inner: ArrayVec<T, N>,
}

impl<T, const N: usize> SetOf<T, N>
where
    T: DerOrd,
{
    /// Create a new [`SetOf`].
    pub fn new() -> Self {
        Self {
            inner: ArrayVec::default(),
        }
    }

    /// Add an item to this [`SetOf`].
    ///
    /// Items MUST be added in lexicographical order according to the
    /// [`DerOrd`] impl on `T`.
    #[deprecated(since = "0.7.6", note = "use `insert` or `insert_ordered` instead")]
    pub fn add(&mut self, new_elem: T) -> Result<()> {
        self.insert_ordered(new_elem)
    }

    /// Insert an item into this [`SetOf`].
    pub fn insert(&mut self, item: T) -> Result<()> {
        self.inner.push(item)?;
        der_sort(self.inner.as_mut())
    }

    /// Insert an item into this [`SetOf`].
    ///
    /// Items MUST be added in lexicographical order according to the
    /// [`DerOrd`] impl on `T`.
    pub fn insert_ordered(&mut self, item: T) -> Result<()> {
        // Ensure set elements are lexicographically ordered
        if let Some(last) = self.inner.last() {
            check_der_ordering(last, &item)?;
        }

        self.inner.push(item)
    }

    /// Get the nth element from this [`SetOf`].
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    /// Iterate over the elements of this [`SetOf`].
    pub fn iter(&self) -> SetOfIter<'_, T> {
        SetOfIter {
            inner: self.inner.iter(),
        }
    }

    /// Is this [`SetOf`] empty?
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of elements in this [`SetOf`].
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T, const N: usize> Default for SetOf<T, N>
where
    T: DerOrd,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T, const N: usize> DecodeValue<'a> for SetOf<T, N>
where
    T: Decode<'a> + DerOrd,
{
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        reader.read_nested(header.length, |reader| {
            let mut result = Self::new();

            while !reader.is_finished() {
                result.inner.push(T::decode(reader)?)?;
            }

            der_sort(result.inner.as_mut())?;
            validate(result.inner.as_ref())?;
            Ok(result)
        })
    }
}

impl<'a, T, const N: usize> EncodeValue for SetOf<T, N>
where
    T: 'a + Decode<'a> + Encode + DerOrd,
{
    fn value_len(&self) -> Result<Length> {
        self.iter()
            .fold(Ok(Length::ZERO), |len, elem| len + elem.encoded_len()?)
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        for elem in self.iter() {
            elem.encode(writer)?;
        }

        Ok(())
    }
}

impl<'a, T, const N: usize> FixedTag for SetOf<T, N>
where
    T: Decode<'a> + DerOrd,
{
    const TAG: Tag = Tag::Set;
}

impl<T, const N: usize> TryFrom<[T; N]> for SetOf<T, N>
where
    T: DerOrd,
{
    type Error = Error;

    fn try_from(mut arr: [T; N]) -> Result<SetOf<T, N>> {
        der_sort(&mut arr)?;

        let mut result = SetOf::new();

        for elem in arr {
            result.insert_ordered(elem)?;
        }

        Ok(result)
    }
}

impl<T, const N: usize> ValueOrd for SetOf<T, N>
where
    T: DerOrd,
{
    fn value_cmp(&self, other: &Self) -> Result<Ordering> {
        iter_cmp(self.iter(), other.iter())
    }
}

/// Iterator over the elements of an [`SetOf`].
#[derive(Clone, Debug)]
pub struct SetOfIter<'a, T> {
    /// Inner iterator.
    inner: arrayvec::Iter<'a, T>,
}

impl<'a, T> Iterator for SetOfIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.inner.next()
    }
}

impl<'a, T> ExactSizeIterator for SetOfIter<'a, T> {}

/// ASN.1 `SET OF` backed by a [`Vec`].
///
/// This type implements an append-only `SET OF` type which is heap-backed
/// and depends on `alloc` support.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct SetOfVec<T>
where
    T: DerOrd,
{
    inner: Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T: DerOrd> Default for SetOfVec<T> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

#[cfg(feature = "alloc")]
impl<T> SetOfVec<T>
where
    T: DerOrd,
{
    /// Create a new [`SetOfVec`].
    pub fn new() -> Self {
        Self {
            inner: Vec::default(),
        }
    }

    /// Create a new [`SetOfVec`] from the given iterator.
    ///
    /// Note: this is an inherent method instead of an impl of the
    /// [`FromIterator`] trait in order to be fallible.
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I>(iter: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
    {
        Vec::from_iter(iter).try_into()
    }

    /// Add an element to this [`SetOfVec`].
    ///
    /// Items MUST be added in lexicographical order according to the
    /// [`DerOrd`] impl on `T`.
    #[deprecated(since = "0.7.6", note = "use `insert` or `insert_ordered` instead")]
    pub fn add(&mut self, item: T) -> Result<()> {
        self.insert_ordered(item)
    }

    /// Extend a [`SetOfVec`] using an iterator.
    ///
    /// Note: this is an inherent method instead of an impl of the
    /// [`Extend`] trait in order to be fallible.
    pub fn extend<I>(&mut self, iter: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
    {
        self.inner.extend(iter);
        der_sort(&mut self.inner)
    }

    /// Insert an item into this [`SetOfVec`]. Must be unique.
    pub fn insert(&mut self, item: T) -> Result<()> {
        self.inner.push(item);
        der_sort(&mut self.inner)
    }

    /// Insert an item into this [`SetOfVec`]. Must be unique.
    ///
    /// Items MUST be added in lexicographical order according to the
    /// [`DerOrd`] impl on `T`.
    pub fn insert_ordered(&mut self, item: T) -> Result<()> {
        // Ensure set elements are lexicographically ordered
        if let Some(last) = self.inner.last() {
            check_der_ordering(last, &item)?;
        }

        self.inner.push(item);
        Ok(())
    }

    /// Borrow the elements of this [`SetOfVec`] as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// Get the nth element from this [`SetOfVec`].
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    /// Convert this [`SetOfVec`] into the inner [`Vec`].
    pub fn into_vec(self) -> Vec<T> {
        self.inner
    }

    /// Iterate over the elements of this [`SetOfVec`].
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.inner.iter()
    }

    /// Is this [`SetOfVec`] empty?
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of elements in this [`SetOfVec`].
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(feature = "alloc")]
impl<T> AsRef<[T]> for SetOfVec<T>
where
    T: DerOrd,
{
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

#[cfg(feature = "alloc")]
impl<'a, T> DecodeValue<'a> for SetOfVec<T>
where
    T: Decode<'a> + DerOrd,
{
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        reader.read_nested(header.length, |reader| {
            let mut inner = Vec::new();

            while !reader.is_finished() {
                inner.push(T::decode(reader)?);
            }

            der_sort(inner.as_mut())?;
            validate(inner.as_ref())?;
            Ok(Self { inner })
        })
    }
}

#[cfg(feature = "alloc")]
impl<'a, T> EncodeValue for SetOfVec<T>
where
    T: 'a + Decode<'a> + Encode + DerOrd,
{
    fn value_len(&self) -> Result<Length> {
        self.iter()
            .fold(Ok(Length::ZERO), |len, elem| len + elem.encoded_len()?)
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        for elem in self.iter() {
            elem.encode(writer)?;
        }

        Ok(())
    }
}

#[cfg(feature = "alloc")]
impl<T> FixedTag for SetOfVec<T>
where
    T: DerOrd,
{
    const TAG: Tag = Tag::Set;
}

#[cfg(feature = "alloc")]
impl<T> From<SetOfVec<T>> for Vec<T>
where
    T: DerOrd,
{
    fn from(set: SetOfVec<T>) -> Vec<T> {
        set.into_vec()
    }
}

#[cfg(feature = "alloc")]
impl<T> TryFrom<Vec<T>> for SetOfVec<T>
where
    T: DerOrd,
{
    type Error = Error;

    fn try_from(mut vec: Vec<T>) -> Result<SetOfVec<T>> {
        der_sort(vec.as_mut_slice())?;
        Ok(SetOfVec { inner: vec })
    }
}

#[cfg(feature = "alloc")]
impl<T, const N: usize> TryFrom<[T; N]> for SetOfVec<T>
where
    T: DerOrd,
{
    type Error = Error;

    fn try_from(arr: [T; N]) -> Result<SetOfVec<T>> {
        Vec::from(arr).try_into()
    }
}

#[cfg(feature = "alloc")]
impl<T> ValueOrd for SetOfVec<T>
where
    T: DerOrd,
{
    fn value_cmp(&self, other: &Self) -> Result<Ordering> {
        iter_cmp(self.iter(), other.iter())
    }
}

// Implement by hand because the derive would create invalid values.
// Use the conversion from Vec to create a valid value.
#[cfg(feature = "arbitrary")]
impl<'a, T> arbitrary::Arbitrary<'a> for SetOfVec<T>
where
    T: DerOrd + arbitrary::Arbitrary<'a>,
{
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Self::try_from(
            u.arbitrary_iter()?
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
        .map_err(|_| arbitrary::Error::IncorrectFormat)
    }

    fn size_hint(_depth: usize) -> (usize, Option<usize>) {
        (0, None)
    }
}

/// Ensure set elements are lexicographically ordered using [`DerOrd`].
fn check_der_ordering<T: DerOrd>(a: &T, b: &T) -> Result<()> {
    match a.der_cmp(b)? {
        Ordering::Less => Ok(()),
        Ordering::Equal => Err(ErrorKind::SetDuplicate.into()),
        Ordering::Greater => Err(ErrorKind::SetOrdering.into()),
    }
}

/// Sort a mut slice according to its [`DerOrd`], returning any errors which
/// might occur during the comparison.
///
/// The algorithm is insertion sort, which should perform well when the input
/// is mostly sorted to begin with.
///
/// This function is used rather than Rust's built-in `[T]::sort_by` in order
/// to support heapless `no_std` targets as well as to enable bubbling up
/// sorting errors.
#[allow(clippy::integer_arithmetic)]
fn der_sort<T: DerOrd>(slice: &mut [T]) -> Result<()> {
    for i in 0..slice.len() {
        let mut j = i;

        while j > 0 {
            match slice[j - 1].der_cmp(&slice[j])? {
                Ordering::Less => break,
                Ordering::Equal => return Err(ErrorKind::SetDuplicate.into()),
                Ordering::Greater => {
                    slice.swap(j - 1, j);
                    j -= 1;
                }
            }
        }
    }

    Ok(())
}

/// Validate the elements of a `SET OF`, ensuring that they are all in order
/// and that there are no duplicates.
fn validate<T: DerOrd>(slice: &[T]) -> Result<()> {
    if let Some(len) = slice.len().checked_sub(1) {
        for i in 0..len {
            let j = i.checked_add(1).ok_or(ErrorKind::Overflow)?;

            match slice.get(i..=j) {
                Some([a, b]) => {
                    if a.der_cmp(b)? != Ordering::Less {
                        return Err(ErrorKind::SetOrdering.into());
                    }
                }
                _ => return Err(Tag::Set.value_error()),
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::SetOf;
    #[cfg(feature = "alloc")]
    use super::SetOfVec;
    use crate::ErrorKind;

    #[test]
    fn setof_tryfrom_array() {
        let arr = [3u16, 2, 1, 65535, 0];
        let set = SetOf::try_from(arr).unwrap();
        assert!(set.iter().copied().eq([0, 1, 2, 3, 65535]));
    }

    #[test]
    fn setof_tryfrom_array_reject_duplicates() {
        let arr = [1u16, 1];
        let err = SetOf::try_from(arr).err().unwrap();
        assert_eq!(err.kind(), ErrorKind::SetDuplicate);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn setofvec_tryfrom_array() {
        let arr = [3u16, 2, 1, 65535, 0];
        let set = SetOfVec::try_from(arr).unwrap();
        assert_eq!(set.as_ref(), &[0, 1, 2, 3, 65535]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn setofvec_tryfrom_vec() {
        let vec = vec![3u16, 2, 1, 65535, 0];
        let set = SetOfVec::try_from(vec).unwrap();
        assert_eq!(set.as_ref(), &[0, 1, 2, 3, 65535]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn setofvec_tryfrom_vec_reject_duplicates() {
        let vec = vec![1u16, 1];
        let err = SetOfVec::try_from(vec).err().unwrap();
        assert_eq!(err.kind(), ErrorKind::SetDuplicate);
    }
}
