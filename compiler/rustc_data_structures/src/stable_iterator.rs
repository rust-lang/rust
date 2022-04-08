use std::iter::Chain;

use crate::stable_hasher::ToStableHashKey;

pub struct StableIterator<I: Iterator> {
    inner: I,
}

impl<T, I: Iterator<Item = T>> StableIterator<I> {
    #[inline]
    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> StableIterator<impl Iterator<Item = U>> {
        StableIterator { inner: self.inner.map(f) }
    }

    #[inline]
    pub fn into_sorted<HCX>(self, hcx: &HCX) -> Vec<T>
    where
        T: ToStableHashKey<HCX>,
    {
        let mut items: Vec<T> = self.inner.collect();
        items.sort_by_cached_key(|x| x.to_stable_hash_key(hcx));
        items
    }

    #[inline]
    pub fn any<F: Fn(T) -> bool>(&mut self, f: F) -> bool {
        self.inner.any(f)
    }

    #[inline]
    pub fn all<F: Fn(T) -> bool>(&mut self, f: F) -> bool {
        self.inner.all(f)
    }

    #[inline]
    pub fn chain<J: Iterator<Item = I::Item>>(self, other: StableIterator<J>) -> StableChain<I, J> {
        self.inner.chain(other.inner).into()
    }
}

pub trait IntoStableIterator {
    type IntoIter: Iterator;
    fn into_stable_iter(self) -> StableIterator<Self::IntoIter>;
}

impl<I: Iterator, S: IntoIterator<Item = I::Item, IntoIter = I>> IntoStableIterator for S {
    type IntoIter = I;

    #[inline]
    fn into_stable_iter(self) -> StableIterator<I> {
        StableIterator { inner: self.into_iter() }
    }
}

pub struct StableChain<I: Iterator, J: Iterator> {
    inner: Chain<I, J>,
}

impl<I: Iterator, J: Iterator<Item = I::Item>> IntoStableIterator for StableChain<I, J> {
    type IntoIter = Chain<I, J>;

    #[inline]
    fn into_stable_iter(self) -> StableIterator<Self::IntoIter> {
        self.inner.into_stable_iter()
    }
}

impl<I: Iterator, J: Iterator> From<Chain<I, J>> for StableChain<I, J> {
    #[inline]
    fn from(inner: Chain<I, J>) -> Self {
        Self { inner }
    }
}
