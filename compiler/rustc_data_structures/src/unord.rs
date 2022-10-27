//! This module contains collection types that don't expose their internal
//! ordering. This is a useful property for deterministic computations, such
//! as required by the query system.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::{
    borrow::Borrow,
    hash::Hash,
    iter::{Product, Sum},
};

use crate::{
    fingerprint::Fingerprint,
    stable_hasher::{HashStable, StableHasher, ToStableHashKey},
};

/// `UnordItems` is the order-less version of `Iterator`. It only contains methods
/// that don't (easily) expose an ordering of the underlying items.
///
/// Most methods take an `Fn` where the `Iterator`-version takes an `FnMut`. This
/// is to reduce the risk of accidentally leaking the internal order via the closure
/// environment. Otherwise one could easily do something like
///
/// ```rust,ignore (pseudo code)
/// let mut ordered = vec![];
/// unordered_items.all(|x| ordered.push(x));
/// ```
///
/// It's still possible to do the same thing with an `Fn` by using interior mutability,
/// but the chance of doing it accidentally is reduced.
pub struct UnordItems<T, I: Iterator<Item = T>>(I);

impl<T, I: Iterator<Item = T>> UnordItems<T, I> {
    #[inline]
    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> UnordItems<U, impl Iterator<Item = U>> {
        UnordItems(self.0.map(f))
    }

    #[inline]
    pub fn all<U, F: Fn(T) -> bool>(mut self, f: F) -> bool {
        self.0.all(f)
    }

    #[inline]
    pub fn any<U, F: Fn(T) -> bool>(mut self, f: F) -> bool {
        self.0.any(f)
    }

    #[inline]
    pub fn filter<U, F: Fn(&T) -> bool>(self, f: F) -> UnordItems<T, impl Iterator<Item = T>> {
        UnordItems(self.0.filter(f))
    }

    #[inline]
    pub fn filter_map<U, F: Fn(T) -> Option<U>>(
        self,
        f: F,
    ) -> UnordItems<U, impl Iterator<Item = U>> {
        UnordItems(self.0.filter_map(f))
    }

    #[inline]
    pub fn max(self) -> Option<T>
    where
        T: Ord,
    {
        self.0.max()
    }

    #[inline]
    pub fn min(self) -> Option<T>
    where
        T: Ord,
    {
        self.0.min()
    }

    #[inline]
    pub fn sum<S>(self) -> S
    where
        S: Sum<T>,
    {
        self.0.sum()
    }

    #[inline]
    pub fn product<S>(self) -> S
    where
        S: Product<T>,
    {
        self.0.product()
    }

    #[inline]
    pub fn count(self) -> usize {
        self.0.count()
    }
}

impl<'a, T: Clone + 'a, I: Iterator<Item = &'a T>> UnordItems<&'a T, I> {
    #[inline]
    pub fn cloned(self) -> UnordItems<T, impl Iterator<Item = T>> {
        UnordItems(self.0.cloned())
    }
}

impl<'a, T: Copy + 'a, I: Iterator<Item = &'a T>> UnordItems<&'a T, I> {
    #[inline]
    pub fn copied(self) -> UnordItems<T, impl Iterator<Item = T>> {
        UnordItems(self.0.copied())
    }
}

impl<T: Ord, I: Iterator<Item = T>> UnordItems<T, I> {
    pub fn into_sorted<HCX>(self, hcx: &HCX) -> Vec<T>
    where
        T: ToStableHashKey<HCX>,
    {
        let mut items: Vec<T> = self.0.collect();
        items.sort_by_cached_key(|x| x.to_stable_hash_key(hcx));
        items
    }

    pub fn into_sorted_small_vec<HCX, const LEN: usize>(self, hcx: &HCX) -> SmallVec<[T; LEN]>
    where
        T: ToStableHashKey<HCX>,
    {
        let mut items: SmallVec<[T; LEN]> = self.0.collect();
        items.sort_by_cached_key(|x| x.to_stable_hash_key(hcx));
        items
    }
}

/// This is a set collection type that tries very hard to not expose
/// any internal iteration. This is a useful property when trying to
/// uphold the determinism invariants imposed by the query system.
///
/// This collection type is a good choice for set-like collections the
/// keys of which don't have a semantic ordering.
///
/// See [MCP 533](https://github.com/rust-lang/compiler-team/issues/533)
/// for more information.
#[derive(Debug, Eq, PartialEq, Clone, Encodable, Decodable)]
pub struct UnordSet<V: Eq + Hash> {
    inner: FxHashSet<V>,
}

impl<V: Eq + Hash> Default for UnordSet<V> {
    fn default() -> Self {
        Self { inner: FxHashSet::default() }
    }
}

impl<V: Eq + Hash> UnordSet<V> {
    #[inline]
    pub fn new() -> Self {
        Self { inner: Default::default() }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn insert(&mut self, v: V) -> bool {
        self.inner.insert(v)
    }

    #[inline]
    pub fn contains<Q: ?Sized>(&self, v: &Q) -> bool
    where
        V: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.contains(v)
    }

    #[inline]
    pub fn items<'a>(&'a self) -> UnordItems<&'a V, impl Iterator<Item = &'a V>> {
        UnordItems(self.inner.iter())
    }

    #[inline]
    pub fn into_items(self) -> UnordItems<V, impl Iterator<Item = V>> {
        UnordItems(self.inner.into_iter())
    }

    // We can safely extend this UnordSet from a set of unordered values because that
    // won't expose the internal ordering anywhere.
    #[inline]
    pub fn extend<I: Iterator<Item = V>>(&mut self, items: UnordItems<V, I>) {
        self.inner.extend(items.0)
    }
}

impl<V: Hash + Eq> Extend<V> for UnordSet<V> {
    fn extend<T: IntoIterator<Item = V>>(&mut self, iter: T) {
        self.inner.extend(iter)
    }
}

impl<HCX, V: Hash + Eq + HashStable<HCX>> HashStable<HCX> for UnordSet<V> {
    #[inline]
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        hash_iter_order_independent(self.inner.iter(), hcx, hasher);
    }
}

/// This is a map collection type that tries very hard to not expose
/// any internal iteration. This is a useful property when trying to
/// uphold the determinism invariants imposed by the query system.
///
/// This collection type is a good choice for map-like collections the
/// keys of which don't have a semantic ordering.
///
/// See [MCP 533](https://github.com/rust-lang/compiler-team/issues/533)
/// for more information.
#[derive(Debug, Eq, PartialEq, Clone, Encodable, Decodable)]
pub struct UnordMap<K: Eq + Hash, V> {
    inner: FxHashMap<K, V>,
}

impl<K: Eq + Hash, V> Default for UnordMap<K, V> {
    fn default() -> Self {
        Self { inner: FxHashMap::default() }
    }
}

impl<K: Hash + Eq, V> Extend<(K, V)> for UnordMap<K, V> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        self.inner.extend(iter)
    }
}

impl<K: Eq + Hash, V> UnordMap<K, V> {
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.inner.insert(k, v)
    }

    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.contains_key(k)
    }

    #[inline]
    pub fn items<'a>(&'a self) -> UnordItems<(&'a K, &'a V), impl Iterator<Item = (&'a K, &'a V)>> {
        UnordItems(self.inner.iter())
    }

    #[inline]
    pub fn into_items(self) -> UnordItems<(K, V), impl Iterator<Item = (K, V)>> {
        UnordItems(self.inner.into_iter())
    }

    // We can safely extend this UnordMap from a set of unordered values because that
    // won't expose the internal ordering anywhere.
    #[inline]
    pub fn extend<I: Iterator<Item = (K, V)>>(&mut self, items: UnordItems<(K, V), I>) {
        self.inner.extend(items.0)
    }
}

impl<HCX, K: Hash + Eq + HashStable<HCX>, V: HashStable<HCX>> HashStable<HCX> for UnordMap<K, V> {
    #[inline]
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        hash_iter_order_independent(self.inner.iter(), hcx, hasher);
    }
}

/// This is a collection type that tries very hard to not expose
/// any internal iteration. This is a useful property when trying to
/// uphold the determinism invariants imposed by the query system.
///
/// This collection type is a good choice for collections the
/// keys of which don't have a semantic ordering and don't implement
/// `Hash` or `Eq`.
///
/// See [MCP 533](https://github.com/rust-lang/compiler-team/issues/533)
/// for more information.
#[derive(Default, Debug, Eq, PartialEq, Clone, Encodable, Decodable)]
pub struct UnordBag<V> {
    inner: Vec<V>,
}

impl<V> UnordBag<V> {
    #[inline]
    pub fn new() -> Self {
        Self { inner: Default::default() }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn push(&mut self, v: V) {
        self.inner.push(v);
    }

    #[inline]
    pub fn items<'a>(&'a self) -> UnordItems<&'a V, impl Iterator<Item = &'a V>> {
        UnordItems(self.inner.iter())
    }

    #[inline]
    pub fn into_items(self) -> UnordItems<V, impl Iterator<Item = V>> {
        UnordItems(self.inner.into_iter())
    }

    // We can safely extend this UnordSet from a set of unordered values because that
    // won't expose the internal ordering anywhere.
    #[inline]
    pub fn extend<I: Iterator<Item = V>>(&mut self, items: UnordItems<V, I>) {
        self.inner.extend(items.0)
    }
}

impl<T> Extend<T> for UnordBag<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.inner.extend(iter)
    }
}

impl<HCX, V: Hash + Eq + HashStable<HCX>> HashStable<HCX> for UnordBag<V> {
    #[inline]
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        hash_iter_order_independent(self.inner.iter(), hcx, hasher);
    }
}

fn hash_iter_order_independent<
    HCX,
    T: HashStable<HCX>,
    I: Iterator<Item = T> + ExactSizeIterator,
>(
    mut it: I,
    hcx: &mut HCX,
    hasher: &mut StableHasher,
) {
    let len = it.len();
    len.hash_stable(hcx, hasher);

    match len {
        0 => {
            // We're done
        }
        1 => {
            // No need to instantiate a hasher
            it.next().unwrap().hash_stable(hcx, hasher);
        }
        _ => {
            let mut accumulator = Fingerprint::ZERO;
            for item in it {
                let mut item_hasher = StableHasher::new();
                item.hash_stable(hcx, &mut item_hasher);
                let item_fingerprint: Fingerprint = item_hasher.finish();
                accumulator = accumulator.combine_commutative(item_fingerprint);
            }
            accumulator.hash_stable(hcx, hasher);
        }
    }
}

// Do not implement IntoIterator for the collections in this module.
// They only exist to hide iteration order in the first place.
impl<T> !IntoIterator for UnordBag<T> {}
impl<V> !IntoIterator for UnordSet<V> {}
impl<K, V> !IntoIterator for UnordMap<K, V> {}
impl<T, I> !IntoIterator for UnordItems<T, I> {}
