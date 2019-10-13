pub use rustc_hash::FxHashSet;
use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;

/// A deterministic wrapper around FxHashSet that does not provide iteration support.
///
/// It supports insert, remove, get functions from FxHashSet.
/// It also allows to convert hashset to a sorted vector with the method `into_sorted_vector()`.
#[derive(Clone)]
pub struct StableSet<T> {
    base: FxHashSet<T>,
}

impl<T> Default for StableSet<T>
where
    T: Eq + Hash,
{
    fn default() -> StableSet<T> {
        StableSet::new()
    }
}

impl<T> fmt::Debug for StableSet<T>
where
    T: Eq + Hash + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.base)
    }
}

impl<T> PartialEq<StableSet<T>> for StableSet<T>
where
    T: Eq + Hash,
{
    fn eq(&self, other: &StableSet<T>) -> bool {
        self.base == other.base
    }
}

impl<T> Eq for StableSet<T> where T: Eq + Hash {}

impl<T: Hash + Eq> StableSet<T> {
    pub fn new() -> StableSet<T> {
        StableSet { base: FxHashSet::default() }
    }

    pub fn into_sorted_vector(self) -> Vec<T>
    where
        T: Ord,
    {
        let mut vector = self.base.into_iter().collect::<Vec<_>>();
        vector.sort_unstable();
        vector
    }

    pub fn get<Q: ?Sized>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.base.get(value)
    }

    pub fn insert(&mut self, value: T) -> bool {
        self.base.insert(value)
    }

    pub fn remove<Q: ?Sized>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.base.remove(value)
    }
}
