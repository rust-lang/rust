use core::fmt::{self, Debug};

use Entry::*;

use super::{SetValZST, map};
use crate::alloc::{Allocator, Global};

/// A view into a single entry in a set, which may either be vacant or occupied.
///
/// This `enum` is constructed from the [`entry`] method on [`BTreeSet`].
///
/// [`BTreeSet`]: super::BTreeSet
/// [`entry`]: super::BTreeSet::entry
///
/// # Examples
///
/// ```
/// #![feature(btree_set_entry)]
///
/// use std::collections::btree_set::BTreeSet;
///
/// let mut set = BTreeSet::new();
/// set.extend(["a", "b", "c"]);
/// assert_eq!(set.len(), 3);
///
/// // Existing value (insert)
/// let entry = set.entry("a");
/// let _raw_o = entry.insert();
/// assert_eq!(set.len(), 3);
/// // Nonexistent value (insert)
/// set.entry("d").insert();
///
/// // Existing value (or_insert)
/// set.entry("b").or_insert();
/// // Nonexistent value (or_insert)
/// set.entry("e").or_insert();
///
/// println!("Our BTreeSet: {:?}", set);
/// assert!(set.iter().eq(&["a", "b", "c", "d", "e"]));
/// ```
#[unstable(feature = "btree_set_entry", issue = "133549")]
pub enum Entry<
    'a,
    T,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + Clone = Global,
> {
    /// An occupied entry.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::btree_set::{Entry, BTreeSet};
    ///
    /// let mut set = BTreeSet::from(["a", "b"]);
    ///
    /// match set.entry("a") {
    ///     Entry::Vacant(_) => unreachable!(),
    ///     Entry::Occupied(_) => { }
    /// }
    /// ```
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    Occupied(OccupiedEntry<'a, T, A>),

    /// A vacant entry.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::btree_set::{Entry, BTreeSet};
    ///
    /// let mut set = BTreeSet::new();
    ///
    /// match set.entry("a") {
    ///     Entry::Occupied(_) => unreachable!(),
    ///     Entry::Vacant(_) => { }
    /// }
    /// ```
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    Vacant(VacantEntry<'a, T, A>),
}

#[unstable(feature = "btree_set_entry", issue = "133549")]
impl<T: Debug + Ord, A: Allocator + Clone> Debug for Entry<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Vacant(ref v) => f.debug_tuple("Entry").field(v).finish(),
            Occupied(ref o) => f.debug_tuple("Entry").field(o).finish(),
        }
    }
}

/// A view into an occupied entry in a `BTreeSet`.
/// It is part of the [`Entry`] enum.
///
/// # Examples
///
/// ```
/// #![feature(btree_set_entry)]
///
/// use std::collections::btree_set::{Entry, BTreeSet};
///
/// let mut set = BTreeSet::new();
/// set.extend(["a", "b", "c"]);
///
/// let _entry_o = set.entry("a").insert();
/// assert_eq!(set.len(), 3);
///
/// // Existing key
/// match set.entry("a") {
///     Entry::Vacant(_) => unreachable!(),
///     Entry::Occupied(view) => {
///         assert_eq!(view.get(), &"a");
///     }
/// }
///
/// assert_eq!(set.len(), 3);
///
/// // Existing key (take)
/// match set.entry("c") {
///     Entry::Vacant(_) => unreachable!(),
///     Entry::Occupied(view) => {
///         assert_eq!(view.remove(), "c");
///     }
/// }
/// assert_eq!(set.get(&"c"), None);
/// assert_eq!(set.len(), 2);
/// ```
#[unstable(feature = "btree_set_entry", issue = "133549")]
pub struct OccupiedEntry<
    'a,
    T,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + Clone = Global,
> {
    pub(super) inner: map::OccupiedEntry<'a, T, SetValZST, A>,
}

#[unstable(feature = "btree_set_entry", issue = "133549")]
impl<T: Debug + Ord, A: Allocator + Clone> Debug for OccupiedEntry<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry").field("value", self.get()).finish()
    }
}

/// A view into a vacant entry in a `BTreeSet`.
/// It is part of the [`Entry`] enum.
///
/// # Examples
///
/// ```
/// #![feature(btree_set_entry)]
///
/// use std::collections::btree_set::{Entry, BTreeSet};
///
/// let mut set = BTreeSet::<&str>::new();
///
/// let entry_v = match set.entry("a") {
///     Entry::Vacant(view) => view,
///     Entry::Occupied(_) => unreachable!(),
/// };
/// entry_v.insert();
/// assert!(set.contains("a") && set.len() == 1);
///
/// // Nonexistent key (insert)
/// match set.entry("b") {
///     Entry::Vacant(view) => view.insert(),
///     Entry::Occupied(_) => unreachable!(),
/// }
/// assert!(set.contains("b") && set.len() == 2);
/// ```
#[unstable(feature = "btree_set_entry", issue = "133549")]
pub struct VacantEntry<
    'a,
    T,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + Clone = Global,
> {
    pub(super) inner: map::VacantEntry<'a, T, SetValZST, A>,
}

#[unstable(feature = "btree_set_entry", issue = "133549")]
impl<T: Debug + Ord, A: Allocator + Clone> Debug for VacantEntry<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VacantEntry").field(self.get()).finish()
    }
}

impl<'a, T: Ord, A: Allocator + Clone> Entry<'a, T, A> {
    /// Sets the value of the entry, and returns an `OccupiedEntry`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    /// let entry = set.entry("horseyland").insert();
    ///
    /// assert_eq!(entry.get(), &"horseyland");
    /// ```
    #[inline]
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    pub fn insert(self) -> OccupiedEntry<'a, T, A> {
        match self {
            Occupied(entry) => entry,
            Vacant(entry) => entry.insert_entry(),
        }
    }

    /// Ensures a value is in the entry by inserting if it was vacant.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    ///
    /// // nonexistent key
    /// set.entry("poneyland").or_insert();
    /// assert!(set.contains("poneyland"));
    ///
    /// // existing key
    /// set.entry("poneyland").or_insert();
    /// assert!(set.contains("poneyland"));
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    pub fn or_insert(self) {
        if let Vacant(entry) = self {
            entry.insert();
        }
    }

    /// Returns a reference to this entry's value.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    /// set.entry("poneyland").or_insert();
    ///
    /// // existing key
    /// assert_eq!(set.entry("poneyland").get(), &"poneyland");
    /// // nonexistent key
    /// assert_eq!(set.entry("horseland").get(), &"horseland");
    /// ```
    #[inline]
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    pub fn get(&self) -> &T {
        match *self {
            Occupied(ref entry) => entry.get(),
            Vacant(ref entry) => entry.get(),
        }
    }
}

impl<'a, T: Ord, A: Allocator + Clone> OccupiedEntry<'a, T, A> {
    /// Gets a reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::btree_set::{Entry, BTreeSet};
    ///
    /// let mut set = BTreeSet::new();
    /// set.entry("poneyland").or_insert();
    ///
    /// match set.entry("poneyland") {
    ///     Entry::Vacant(_) => panic!(),
    ///     Entry::Occupied(entry) => assert_eq!(entry.get(), &"poneyland"),
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    pub fn get(&self) -> &T {
        self.inner.key()
    }

    /// Takes the value out of the entry, and returns it.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::BTreeSet;
    /// use std::collections::btree_set::Entry;
    ///
    /// let mut set = BTreeSet::new();
    /// set.entry("poneyland").or_insert();
    ///
    /// if let Entry::Occupied(o) = set.entry("poneyland") {
    ///     assert_eq!(o.remove(), "poneyland");
    /// }
    ///
    /// assert_eq!(set.contains("poneyland"), false);
    /// ```
    #[inline]
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    pub fn remove(self) -> T {
        self.inner.remove_entry().0
    }
}

impl<'a, T: Ord, A: Allocator + Clone> VacantEntry<'a, T, A> {
    /// Gets a reference to the value that would be used when inserting
    /// through the `VacantEntry`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    /// assert_eq!(set.entry("poneyland").get(), &"poneyland");
    /// ```
    #[inline]
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    pub fn get(&self) -> &T {
        self.inner.key()
    }

    /// Take ownership of the value.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::btree_set::{Entry, BTreeSet};
    ///
    /// let mut set = BTreeSet::new();
    ///
    /// match set.entry("poneyland") {
    ///     Entry::Occupied(_) => panic!(),
    ///     Entry::Vacant(v) => assert_eq!(v.into_value(), "poneyland"),
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    pub fn into_value(self) -> T {
        self.inner.into_key()
    }

    /// Sets the value of the entry with the VacantEntry's value.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_set_entry)]
    ///
    /// use std::collections::BTreeSet;
    /// use std::collections::btree_set::Entry;
    ///
    /// let mut set = BTreeSet::new();
    ///
    /// if let Entry::Vacant(o) = set.entry("poneyland") {
    ///     o.insert();
    /// }
    /// assert!(set.contains("poneyland"));
    /// ```
    #[inline]
    #[unstable(feature = "btree_set_entry", issue = "133549")]
    pub fn insert(self) {
        self.inner.insert(SetValZST);
    }

    #[inline]
    fn insert_entry(self) -> OccupiedEntry<'a, T, A> {
        OccupiedEntry { inner: self.inner.insert_entry(SetValZST) }
    }
}
