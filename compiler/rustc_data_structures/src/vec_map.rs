use std::borrow::Borrow;
use std::iter::FromIterator;
use std::slice::{Iter, IterMut};
use std::vec::IntoIter;

use crate::stable_hasher::{HashStable, StableHasher};

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct VecMap<K, V>(Vec<(K, V)>);

impl<K, V> VecMap<K, V>
where
    K: PartialEq,
{
    pub fn new() -> Self {
        VecMap(Default::default())
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        if let Some(elem) = self.0.iter_mut().find(|(key, _)| *key == k) {
            Some(std::mem::replace(&mut elem.1, v))
        } else {
            self.0.push((k, v));
            None
        }
    }

    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        self.0.iter().find(|(key, _)| k == key.borrow()).map(|elem| &elem.1)
    }

    pub fn get_by<P>(&self, mut predicate: P) -> Option<&V>
    where
        for<'b> P: FnMut(&'b &(K, V)) -> bool,
    {
        self.0.iter().find(|kv| predicate(kv)).map(|elem| &elem.1)
    }

    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        self.get(k).is_some()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> Iter<'_, (K, V)> {
        self.into_iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, (K, V)> {
        self.into_iter()
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
    type Item = &'a mut (K, V);
    type IntoIter = IterMut<'a, (K, V)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
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

impl<K, V> Extend<(K, V)> for VecMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        self.0.extend(iter);
    }

    fn extend_one(&mut self, item: (K, V)) {
        self.0.extend_one(item);
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
