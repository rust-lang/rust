use super::EitherIter;
use crate::fx::FxHashSet;
use arrayvec::ArrayVec;
use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;
use std::iter::FromIterator;

/// Small-storage-optimized implementation of a set.
///
/// Stores elements in a small array up to a certain length
/// and switches to `HashSet` when that length is exceeded.
///
/// Implements subset of HashSet API.
///
/// Missing HashSet API:
///   all hasher-related
///   try_reserve (unstable)
///   shrink_to (unstable)
///   drain_filter (unstable)
///   get_or_insert/get_or_insert_owned/get_or_insert_with (unstable)
///   difference/symmetric_difference/intersection/union
///   is_disjoint/is_subset/is_superset
///   PartialEq/Eq (requires sorting the array)
///   BitOr/BitAnd/BitXor/Sub
#[derive(Clone)]
pub enum SsoHashSet<T> {
    Array(ArrayVec<[T; 8]>),
    Set(FxHashSet<T>),
}

impl<T> SsoHashSet<T> {
    /// Creates an empty `SsoHashSet`.
    pub fn new() -> Self {
        SsoHashSet::Array(ArrayVec::new())
    }

    /// Creates an empty `SsoHashSet` with the specified capacity.
    pub fn with_capacity(cap: usize) -> Self {
        let array = ArrayVec::new();
        if array.capacity() >= cap {
            SsoHashSet::Array(array)
        } else {
            SsoHashSet::Set(FxHashSet::with_capacity_and_hasher(cap, Default::default()))
        }
    }

    /// Clears the set, removing all values.
    pub fn clear(&mut self) {
        match self {
            SsoHashSet::Array(array) => array.clear(),
            SsoHashSet::Set(set) => set.clear(),
        }
    }

    /// Returns the number of elements the set can hold without reallocating.
    pub fn capacity(&self) -> usize {
        match self {
            SsoHashSet::Array(array) => array.capacity(),
            SsoHashSet::Set(set) => set.capacity(),
        }
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        match self {
            SsoHashSet::Array(array) => array.len(),
            SsoHashSet::Set(set) => set.len(),
        }
    }

    /// Returns `true` if the set contains no elements.
    pub fn is_empty(&self) -> bool {
        match self {
            SsoHashSet::Array(array) => array.is_empty(),
            SsoHashSet::Set(set) => set.is_empty(),
        }
    }

    /// An iterator visiting all elements in arbitrary order.
    /// The iterator element type is `&'a T`.
    pub fn iter(&'a self) -> impl Iterator<Item = &'a T> {
        self.into_iter()
    }

    /// Clears the set, returning all elements in an iterator.
    pub fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        match self {
            SsoHashSet::Array(array) => EitherIter::Left(array.drain(..)),
            SsoHashSet::Set(set) => EitherIter::Right(set.drain()),
        }
    }
}

impl<T: Eq + Hash> SsoHashSet<T> {
    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `SsoHashSet`. The collection may reserve more space to avoid
    /// frequent reallocations.
    pub fn reserve(&mut self, additional: usize) {
        match self {
            SsoHashSet::Array(array) => {
                if array.capacity() < (array.len() + additional) {
                    let mut set: FxHashSet<T> = array.drain(..).collect();
                    set.reserve(additional);
                    *self = SsoHashSet::Set(set);
                }
            }
            SsoHashSet::Set(set) => set.reserve(additional),
        }
    }

    /// Shrinks the capacity of the set as much as possible. It will drop
    /// down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    pub fn shrink_to_fit(&mut self) {
        if let SsoHashSet::Set(set) = self {
            let mut array = ArrayVec::new();
            if set.len() <= array.capacity() {
                array.extend(set.drain());
                *self = SsoHashSet::Array(array);
            } else {
                set.shrink_to_fit();
            }
        }
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        match self {
            SsoHashSet::Array(array) => array.retain(|v| f(v)),
            SsoHashSet::Set(set) => set.retain(f),
        }
    }

    /// Removes and returns the value in the set, if any, that is equal to the given one.
    pub fn take<Q: ?Sized>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        match self {
            SsoHashSet::Array(array) => {
                if let Some(index) = array.iter().position(|val| val.borrow() == value) {
                    Some(array.swap_remove(index))
                } else {
                    None
                }
            }
            SsoHashSet::Set(set) => set.take(value),
        }
    }

    /// Adds a value to the set, replacing the existing value, if any, that is equal to the given
    /// one. Returns the replaced value.
    pub fn replace(&mut self, value: T) -> Option<T> {
        match self {
            SsoHashSet::Array(array) => {
                if let Some(index) = array.iter().position(|val| *val == value) {
                    let old_value = std::mem::replace(&mut array[index], value);
                    Some(old_value)
                } else {
                    None
                }
            }
            SsoHashSet::Set(set) => set.replace(value),
        }
    }

    /// Returns a reference to the value in the set, if any, that is equal to the given value.
    pub fn get<Q: ?Sized>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        match self {
            SsoHashSet::Array(array) => {
                if let Some(index) = array.iter().position(|val| val.borrow() == value) {
                    Some(&array[index])
                } else {
                    None
                }
            }
            SsoHashSet::Set(set) => set.get(value),
        }
    }

    /// Adds a value to the set.
    ///
    /// If the set did not have this value present, `true` is returned.
    ///
    /// If the set did have this value present, `false` is returned.
    pub fn insert(&mut self, elem: T) -> bool {
        match self {
            SsoHashSet::Array(array) => {
                if array.iter().any(|e| *e == elem) {
                    false
                } else {
                    if let Err(error) = array.try_push(elem) {
                        let mut set: FxHashSet<T> = array.drain(..).collect();
                        set.insert(error.element());
                        *self = SsoHashSet::Set(set);
                    }
                    true
                }
            }
            SsoHashSet::Set(set) => set.insert(elem),
        }
    }

    /// Removes a value from the set. Returns whether the value was
    /// present in the set.
    pub fn remove<Q: ?Sized>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        match self {
            SsoHashSet::Array(array) => {
                if let Some(index) = array.iter().position(|val| val.borrow() == value) {
                    array.swap_remove(index);
                    true
                } else {
                    false
                }
            }
            SsoHashSet::Set(set) => set.remove(value),
        }
    }

    /// Returns `true` if the set contains a value.
    pub fn contains<Q: ?Sized>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        match self {
            SsoHashSet::Array(array) => array.iter().any(|v| v.borrow() == value),
            SsoHashSet::Set(set) => set.contains(value),
        }
    }
}

impl<T> Default for SsoHashSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + Hash> FromIterator<T> for SsoHashSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> SsoHashSet<T> {
        let mut set: SsoHashSet<T> = Default::default();
        set.extend(iter);
        set
    }
}

impl<T: Eq + Hash> Extend<T> for SsoHashSet<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for val in iter.into_iter() {
            self.insert(val);
        }
    }

    fn extend_one(&mut self, item: T) {
        self.insert(item);
    }

    fn extend_reserve(&mut self, additional: usize) {
        match self {
            SsoHashSet::Array(array) => {
                if array.capacity() < (array.len() + additional) {
                    let mut set: FxHashSet<T> = array.drain(..).collect();
                    set.extend_reserve(additional);
                    *self = SsoHashSet::Set(set);
                }
            }
            SsoHashSet::Set(set) => set.extend_reserve(additional),
        }
    }
}

impl<'a, T> Extend<&'a T> for SsoHashSet<T>
where
    T: 'a + Eq + Hash + Copy,
{
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }

    fn extend_one(&mut self, &item: &'a T) {
        self.insert(item);
    }

    fn extend_reserve(&mut self, additional: usize) {
        Extend::<T>::extend_reserve(self, additional)
    }
}

impl<T> IntoIterator for SsoHashSet<T> {
    type IntoIter = EitherIter<
        <ArrayVec<[T; 8]> as IntoIterator>::IntoIter,
        <FxHashSet<T> as IntoIterator>::IntoIter,
    >;
    type Item = <Self::IntoIter as Iterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            SsoHashSet::Array(array) => EitherIter::Left(array.into_iter()),
            SsoHashSet::Set(set) => EitherIter::Right(set.into_iter()),
        }
    }
}

impl<'a, T> IntoIterator for &'a SsoHashSet<T> {
    type IntoIter = EitherIter<
        <&'a ArrayVec<[T; 8]> as IntoIterator>::IntoIter,
        <&'a FxHashSet<T> as IntoIterator>::IntoIter,
    >;
    type Item = <Self::IntoIter as Iterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            SsoHashSet::Array(array) => EitherIter::Left(array.into_iter()),
            SsoHashSet::Set(set) => EitherIter::Right(set.into_iter()),
        }
    }
}

impl<T> fmt::Debug for SsoHashSet<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}
