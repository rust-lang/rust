use std::borrow::Borrow;
use std::fmt::Debug;
use std::iter::FromIterator;
use std::slice::Iter;
use std::vec::IntoIter;

use crate::stable_hasher::{HashStable, StableHasher};

/// A map type implemented as a vector of pairs `K` (key) and `V` (value).
/// It currently provides a subset of all the map operations, the rest could be added as needed.
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct VecMap<K, V>(Vec<(K, V)>);

impl<K, V> VecMap<K, V>
where
    K: Debug + PartialEq,
    V: Debug,
{
    pub fn new() -> Self {
        VecMap(Default::default())
    }

    /// Sets the value of the entry, and returns the entry's old value.
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        if let Some(elem) = self.0.iter_mut().find(|(key, _)| *key == k) {
            Some(std::mem::replace(&mut elem.1, v))
        } else {
            self.0.push((k, v));
            None
        }
    }

    /// Gets a reference to the value in the entry.
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        self.0.iter().find(|(key, _)| k == key.borrow()).map(|elem| &elem.1)
    }

    /// Returns the any value corresponding to the supplied predicate filter.
    ///
    /// The supplied predicate will be applied to each (key, value) pair and it will return a
    /// reference to the values where the predicate returns `true`.
    pub fn any_value_matching(&self, mut predicate: impl FnMut(&(K, V)) -> bool) -> Option<&V> {
        self.0.iter().find(|kv| predicate(kv)).map(|elem| &elem.1)
    }

    /// Returns the value corresponding to the supplied predicate filter. It crashes if there's
    /// more than one matching element.
    ///
    /// The supplied predicate will be applied to each (key, value) pair and it will return a
    /// reference to the value where the predicate returns `true`.
    pub fn get_value_matching(&self, mut predicate: impl FnMut(&(K, V)) -> bool) -> Option<&V> {
        let mut filter = self.0.iter().filter(|kv| predicate(kv));
        let (_, value) = filter.next()?;
        // This should return just one element, otherwise it's a bug
        assert!(
            filter.next().is_none(),
            "Collection {:?} should have just one matching element",
            self
        );
        Some(value)
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type,
    /// [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        self.get(k).is_some()
    }

    /// Returns `true` if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> Iter<'_, (K, V)> {
        self.into_iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.into_iter()
    }

    pub fn retain(&mut self, f: impl Fn(&(K, V)) -> bool) {
        self.0.retain(f)
    }
}

impl<K, V> Default for VecMap<K, V> {
    #[inline]
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<K, V> From<Vec<(K, V)>> for VecMap<K, V> {
    fn from(vec: Vec<(K, V)>) -> Self {
        Self(vec)
    }
}

impl<K, V> Into<Vec<(K, V)>> for VecMap<K, V> {
    fn into(self) -> Vec<(K, V)> {
        self.0
    }
}

impl<K, V> FromIterator<(K, V)> for VecMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<'a, K, V> IntoIterator for &'a VecMap<K, V> {
    type Item = &'a (K, V);
    type IntoIter = Iter<'a, (K, V)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut VecMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = impl Iterator<Item = Self::Item>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut().map(|(k, v)| (&*k, v))
    }
}

impl<K, V> IntoIterator for VecMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<(K, V)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<K: PartialEq + Debug, V: Debug> Extend<(K, V)> for VecMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }

    fn extend_one(&mut self, (k, v): (K, V)) {
        self.insert(k, v);
    }

    fn extend_reserve(&mut self, additional: usize) {
        self.0.extend_reserve(additional);
    }
}

impl<K, V, CTX> HashStable<CTX> for VecMap<K, V>
where
    K: HashStable<CTX> + Eq,
    V: HashStable<CTX>,
{
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.0.hash_stable(hcx, hasher)
    }
}

#[cfg(test)]
mod tests;
