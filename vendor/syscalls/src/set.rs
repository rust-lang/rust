//! Enables the creation of a syscall bitset.

use super::Sysno;

use core::fmt;
use core::num::NonZeroUsize;

const fn bits_per<T>() -> usize {
    core::mem::size_of::<T>().saturating_mul(8)
}

/// Returns the number of words of type `T` required to hold the specified
/// number of `bits`.
const fn words<T>(bits: usize) -> usize {
    let width = bits_per::<T>();
    if width == 0 {
        return 0;
    }

    bits / width + ((bits % width != 0) as usize)
}

/// A set of syscalls.
///
/// This provides constant-time lookup of syscalls within a bitset.
///
/// # Examples
///
/// ```
/// # use syscalls::{Sysno, SysnoSet};
/// let syscalls = SysnoSet::new(&[Sysno::read, Sysno::write, Sysno::openat, Sysno::close]);
/// assert!(syscalls.contains(Sysno::read));
/// assert!(syscalls.contains(Sysno::close));
/// ```
/// Most operations can be done at compile-time as well.
/// ```
/// # use syscalls::{Sysno, SysnoSet};
/// const SYSCALLS: SysnoSet =
///     SysnoSet::new(&[Sysno::read, Sysno::write, Sysno::close])
///         .union(&SysnoSet::new(&[Sysno::openat]));
/// const _: () = assert!(SYSCALLS.contains(Sysno::read));
/// const _: () = assert!(SYSCALLS.contains(Sysno::openat));
/// ```
#[derive(Clone, Eq, PartialEq)]
pub struct SysnoSet {
    pub(crate) data: [usize; words::<usize>(Sysno::table_size())],
}

impl Default for SysnoSet {
    fn default() -> Self {
        Self::empty()
    }
}

impl SysnoSet {
    /// The set of all valid syscalls.
    const ALL: &'static Self = &Self::new(Sysno::ALL);

    const WORD_WIDTH: usize = usize::BITS as usize;

    /// Compute the index and mask for the given syscall as stored in the set data.
    #[inline]
    pub(crate) const fn get_idx_mask(sysno: Sysno) -> (usize, usize) {
        let bit = (sysno.id() as usize) - (Sysno::first().id() as usize);
        (bit / Self::WORD_WIDTH, 1 << (bit % Self::WORD_WIDTH))
    }

    /// Initialize the syscall set with the given slice of syscalls.
    ///
    /// Since this is a `const fn`, this can be used at compile-time.
    pub const fn new(syscalls: &[Sysno]) -> Self {
        let mut set = Self::empty();

        // Use while-loop because for-loops are not yet allowed in const-fns.
        // https://github.com/rust-lang/rust/issues/87575
        let mut i = 0;
        while i < syscalls.len() {
            let (idx, mask) = Self::get_idx_mask(syscalls[i]);
            set.data[idx] |= mask;
            i += 1;
        }

        set
    }

    /// Creates an empty set of syscalls.
    pub const fn empty() -> Self {
        Self {
            data: [0; words::<usize>(Sysno::table_size())],
        }
    }

    /// Creates a set containing all valid syscalls.
    pub const fn all() -> Self {
        Self {
            data: Self::ALL.data,
        }
    }

    /// Returns true if the set contains the given syscall.
    pub const fn contains(&self, sysno: Sysno) -> bool {
        let (idx, mask) = Self::get_idx_mask(sysno);
        self.data[idx] & mask != 0
    }

    /// Returns true if the set is empty. Although this is an O(1) operation
    /// (because the total number of possible syscalls is always constant), it
    /// must go through the whole bit set to count the number of bits. Thus,
    /// this may have a large, constant overhead.
    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|&x| x == 0)
    }

    /// Clears the set, removing all syscalls.
    pub fn clear(&mut self) {
        for word in &mut self.data {
            *word = 0;
        }
    }

    /// Returns the number of syscalls in the set. Although this is an O(1)
    /// operation (because the total number of syscalls is always constant), it
    /// must go through the whole bit set to count the number of bits. Thus,
    /// this may have a large, constant overhead.
    pub fn count(&self) -> usize {
        self.data
            .iter()
            .fold(0, |acc, x| acc + x.count_ones() as usize)
    }

    /// Inserts the given syscall into the set. Returns true if the syscall was
    /// not already in the set.
    pub fn insert(&mut self, sysno: Sysno) -> bool {
        // The returned value computation will be optimized away by the compiler
        // if not needed.
        let (idx, mask) = Self::get_idx_mask(sysno);
        let old_value = self.data[idx] & mask;
        self.data[idx] |= mask;
        old_value == 0
    }

    /// Removes the given syscall from the set. Returns true if the syscall was
    /// in the set.
    pub fn remove(&mut self, sysno: Sysno) -> bool {
        // The returned value computation will be optimized away by the compiler
        // if not needed.
        let (idx, mask) = Self::get_idx_mask(sysno);
        let old_value = self.data[idx] & mask;
        self.data[idx] &= !mask;
        old_value != 0
    }

    /// Does a set union with this set and another.
    #[must_use]
    pub const fn union(mut self, other: &Self) -> Self {
        let mut i = 0;
        let n = self.data.len();
        while i < n {
            self.data[i] |= other.data[i];
            i += 1;
        }

        self
    }

    /// Does a set intersection with this set and another.
    #[must_use]
    pub const fn intersection(mut self, other: &Self) -> Self {
        let mut i = 0;
        let n = self.data.len();
        while i < n {
            self.data[i] &= other.data[i];
            i += 1;
        }

        self
    }

    /// Calculates the difference with this set and another. That is, the
    /// resulting set only includes the syscalls that are in `self` but not in
    /// `other`.
    #[must_use]
    pub const fn difference(mut self, other: &Self) -> Self {
        let mut i = 0;
        let n = self.data.len();
        while i < n {
            self.data[i] &= !other.data[i];
            i += 1;
        }

        self
    }

    /// Calculates the symmetric difference with this set and another. That is,
    /// the resulting set only includes the syscalls that are in `self` or in
    /// `other`, but not in both.
    #[must_use]
    pub const fn symmetric_difference(mut self, other: &Self) -> Self {
        let mut i = 0;
        let n = self.data.len();
        while i < n {
            self.data[i] ^= other.data[i];
            i += 1;
        }

        self
    }

    /// Returns an iterator that iterates over the syscalls contained in the set.
    pub fn iter(&self) -> SysnoSetIter {
        SysnoSetIter::new(self.data.iter())
    }
}

impl fmt::Debug for SysnoSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl core::ops::BitOr for SysnoSet {
    type Output = Self;

    fn bitor(mut self, rhs: Self) -> Self::Output {
        self |= rhs;
        self
    }
}

impl core::ops::BitOrAssign<&Self> for SysnoSet {
    fn bitor_assign(&mut self, rhs: &Self) {
        for (left, right) in self.data.iter_mut().zip(rhs.data.iter()) {
            *left |= right;
        }
    }
}

impl core::ops::BitOrAssign for SysnoSet {
    fn bitor_assign(&mut self, rhs: Self) {
        *self |= &rhs;
    }
}

impl core::ops::BitOrAssign<Sysno> for SysnoSet {
    fn bitor_assign(&mut self, sysno: Sysno) {
        self.insert(sysno);
    }
}

impl FromIterator<Sysno> for SysnoSet {
    fn from_iter<I: IntoIterator<Item = Sysno>>(iter: I) -> Self {
        let mut set = SysnoSet::empty();
        set.extend(iter);
        set
    }
}

impl Extend<Sysno> for SysnoSet {
    fn extend<T: IntoIterator<Item = Sysno>>(&mut self, iter: T) {
        for sysno in iter {
            self.insert(sysno);
        }
    }
}

impl<'a> IntoIterator for &'a SysnoSet {
    type Item = Sysno;
    type IntoIter = SysnoSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Helper for iterating over the non-zero values of the words in the bitset.
struct NonZeroUsizeIter<'a> {
    iter: core::slice::Iter<'a, usize>,
    count: usize,
}

impl<'a> NonZeroUsizeIter<'a> {
    pub fn new(iter: core::slice::Iter<'a, usize>) -> Self {
        Self { iter, count: 0 }
    }
}

impl<'a> Iterator for NonZeroUsizeIter<'a> {
    type Item = NonZeroUsize;

    fn next(&mut self) -> Option<Self::Item> {
        for item in &mut self.iter {
            self.count += 1;

            if let Some(item) = NonZeroUsize::new(*item) {
                return Some(item);
            }
        }

        None
    }
}

/// An iterator over the syscalls contained in a [`SysnoSet`].
pub struct SysnoSetIter<'a> {
    // Our iterator over nonzero words in the bitset.
    iter: NonZeroUsizeIter<'a>,

    // The current word in the set we're operating on. This is only None if the
    // iterator has been exhausted. The next bit that is set is found by
    // counting the number of leading zeros. When found, we just mask it off.
    current: Option<NonZeroUsize>,
}

impl<'a> SysnoSetIter<'a> {
    fn new(iter: core::slice::Iter<'a, usize>) -> Self {
        let mut iter = NonZeroUsizeIter::new(iter);
        let current = iter.next();
        Self { iter, current }
    }
}

impl<'a> Iterator for SysnoSetIter<'a> {
    type Item = Sysno;

    fn next(&mut self) -> Option<Self::Item> {
        // Construct a mask where all but the last bit is set. This is then
        // shifted to remove the first bit we find.
        const MASK: usize = !1usize;

        if let Some(word) = self.current.take() {
            let index = self.iter.count.wrapping_sub(1);

            // Get the index of the next bit. For example:
            //      0b0000000010000
            //                ^
            // Here, there are 4 trailing zeros, so 4 is the next set bit. Since
            // we're only iterating over non-zero words, we are guaranteed to
            // get a valid index.
            let bit = word.trailing_zeros();

            // Mask off that bit and store the resulting word for next time.
            let next_word =
                NonZeroUsize::new(word.get() & MASK.rotate_left(bit));

            self.current = next_word.or_else(|| self.iter.next());

            let offset = Sysno::first().id() as u32;
            let sysno = index as u32 * usize::BITS + bit + offset;

            // TODO: Use an unchecked conversion to speed this up.
            return Some(Sysno::from(sysno));
        }

        None
    }
}

#[cfg(feature = "serde")]
use serde::{
    de::{Deserialize, Deserializer, SeqAccess, Visitor},
    ser::{Serialize, SerializeSeq, Serializer},
};

#[cfg(feature = "serde")]
impl Serialize for SysnoSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.count()))?;
        for sysno in self {
            seq.serialize_element(&sysno)?;
        }
        seq.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for SysnoSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SeqVisitor;

        impl<'de> Visitor<'de> for SeqVisitor {
            type Value = SysnoSet;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a sequence")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = SysnoSet::empty();

                while let Some(value) = seq.next_element()? {
                    values.insert(value);
                }

                Ok(values)
            }
        }

        deserializer.deserialize_seq(SeqVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_words() {
        assert_eq!(words::<u64>(42), 1);
        assert_eq!(words::<u64>(0), 0);
        assert_eq!(words::<u32>(42), 2);
        assert_eq!(words::<()>(42), 0);
    }

    #[test]
    fn test_bits_per() {
        assert_eq!(bits_per::<()>(), 0);
        assert_eq!(bits_per::<u8>(), 8);
        assert_eq!(bits_per::<u32>(), 32);
        assert_eq!(bits_per::<u64>(), 64);
    }

    #[test]
    fn test_default() {
        assert_eq!(SysnoSet::default(), SysnoSet::empty());
    }

    #[test]
    fn test_const_new() {
        static SYSCALLS: SysnoSet =
            SysnoSet::new(&[Sysno::openat, Sysno::read, Sysno::close]);

        assert!(SYSCALLS.contains(Sysno::openat));
        assert!(SYSCALLS.contains(Sysno::read));
        assert!(SYSCALLS.contains(Sysno::close));
        assert!(!SYSCALLS.contains(Sysno::write));
    }

    #[test]
    fn test_contains() {
        let set = SysnoSet::empty();
        assert!(!set.contains(Sysno::openat));
        assert!(!set.contains(Sysno::first()));
        assert!(!set.contains(Sysno::last()));

        let set = SysnoSet::all();
        assert!(set.contains(Sysno::openat));
        assert!(set.contains(Sysno::first()));
        assert!(set.contains(Sysno::last()));
    }

    #[test]
    fn test_is_empty() {
        let mut set = SysnoSet::empty();
        assert!(set.is_empty());
        assert!(set.insert(Sysno::openat));
        assert!(!set.is_empty());
        assert!(set.remove(Sysno::openat));
        assert!(set.is_empty());
        assert!(set.insert(Sysno::last()));
        assert!(!set.is_empty());
    }

    #[test]
    fn test_count() {
        let mut set = SysnoSet::empty();
        assert_eq!(set.count(), 0);
        assert!(set.insert(Sysno::openat));
        assert!(set.insert(Sysno::last()));
        assert_eq!(set.count(), 2);
    }

    #[test]
    fn test_insert() {
        let mut set = SysnoSet::empty();
        assert!(set.insert(Sysno::openat));
        assert!(set.insert(Sysno::read));
        assert!(set.insert(Sysno::close));
        assert!(set.contains(Sysno::openat));
        assert!(set.contains(Sysno::read));
        assert!(set.contains(Sysno::close));
        assert_eq!(set.count(), 3);
    }

    #[test]
    fn test_remove() {
        let mut set = SysnoSet::all();
        assert!(set.remove(Sysno::openat));
        assert!(!set.contains(Sysno::openat));
        assert!(set.contains(Sysno::close));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_from_iter() {
        let set =
            SysnoSet::from_iter(vec![Sysno::openat, Sysno::read, Sysno::close]);
        assert!(set.contains(Sysno::openat));
        assert!(set.contains(Sysno::read));
        assert!(set.contains(Sysno::close));
        assert_eq!(set.count(), 3);
    }

    #[test]
    fn test_all() {
        let mut all = SysnoSet::all();
        assert_eq!(all.count(), Sysno::count());

        all.contains(Sysno::openat);
        all.contains(Sysno::first());
        all.contains(Sysno::last());

        all.clear();

        assert_eq!(all.count(), 0);
    }

    #[test]
    fn test_union() {
        let a = SysnoSet::new(&[Sysno::read, Sysno::openat, Sysno::close]);
        let b = SysnoSet::new(&[Sysno::write, Sysno::openat, Sysno::close]);
        assert_eq!(
            a.union(&b),
            SysnoSet::new(&[
                Sysno::read,
                Sysno::write,
                Sysno::openat,
                Sysno::close
            ])
        );
    }

    #[test]
    fn test_bitorassign() {
        let mut a = SysnoSet::new(&[Sysno::read, Sysno::openat, Sysno::close]);
        let b = SysnoSet::new(&[Sysno::write, Sysno::openat, Sysno::close]);
        a |= &b;
        a |= b;
        a |= Sysno::openat;

        assert_eq!(
            a,
            SysnoSet::new(&[
                Sysno::read,
                Sysno::write,
                Sysno::close,
                Sysno::openat,
            ])
        );
    }

    #[test]
    fn test_bitor() {
        let a = SysnoSet::new(&[Sysno::read, Sysno::openat, Sysno::close]);
        let b = SysnoSet::new(&[Sysno::write, Sysno::openat, Sysno::close]);
        assert_eq!(
            a | b,
            SysnoSet::new(&[
                Sysno::read,
                Sysno::write,
                Sysno::openat,
                Sysno::close,
            ])
        );
    }

    #[test]
    fn test_intersection() {
        let a = SysnoSet::new(&[Sysno::read, Sysno::openat, Sysno::close]);
        let b = SysnoSet::new(&[Sysno::write, Sysno::openat, Sysno::close]);
        assert_eq!(
            a.intersection(&b),
            SysnoSet::new(&[Sysno::openat, Sysno::close])
        );
    }

    #[test]
    fn test_difference() {
        let a = SysnoSet::new(&[Sysno::read, Sysno::openat, Sysno::close]);
        let b = SysnoSet::new(&[Sysno::write, Sysno::openat, Sysno::close]);
        assert_eq!(a.difference(&b), SysnoSet::new(&[Sysno::read]));
    }

    #[test]
    fn test_symmetric_difference() {
        let a = SysnoSet::new(&[Sysno::read, Sysno::openat, Sysno::close]);
        let b = SysnoSet::new(&[Sysno::write, Sysno::openat, Sysno::close]);
        assert_eq!(
            a.symmetric_difference(&b),
            SysnoSet::new(&[Sysno::read, Sysno::write])
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_iter() {
        let syscalls = &[Sysno::read, Sysno::openat, Sysno::close];
        let set = SysnoSet::new(syscalls);

        assert_eq!(set.iter().collect::<Vec<_>>().len(), 3);
    }

    #[test]
    fn test_iter_full() {
        assert_eq!(SysnoSet::all().iter().count(), Sysno::count());
    }

    #[test]
    fn test_into_iter() {
        let syscalls = &[Sysno::read, Sysno::openat, Sysno::close];
        let set = SysnoSet::new(syscalls);

        assert_eq!(set.into_iter().count(), 3);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_debug() {
        let syscalls = &[Sysno::openat, Sysno::read];
        let set = SysnoSet::new(syscalls);
        // The order of the debug output is not guaranteed, so we can't do an exact match
        let result = format!("{:?}", set);
        assert_eq!(result.len(), "{read, openat}".len());
        assert!(result.starts_with('{'));
        assert!(result.ends_with('}'));
        assert!(result.contains(", "));
        assert!(result.contains("read"));
        assert!(result.contains("openat"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_iter_empty() {
        assert_eq!(SysnoSet::empty().iter().collect::<Vec<_>>(), &[]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip() {
        let syscalls = SysnoSet::new(&[
            Sysno::read,
            Sysno::write,
            Sysno::close,
            Sysno::openat,
        ]);

        let s = serde_json::to_string_pretty(&syscalls).unwrap();

        assert_eq!(serde_json::from_str::<SysnoSet>(&s).unwrap(), syscalls);
    }
}
