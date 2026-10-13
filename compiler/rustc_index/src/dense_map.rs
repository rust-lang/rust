use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice::Iter;
use std::vec::IntoIter;

pub trait DenseKey: Copy {
    fn new(i: usize) -> Self;
    fn index(self) -> usize;
}

#[derive(Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct DenseMap<K: DenseKey, V> {
    raw: Vec<V>,
    _marker: PhantomData<K>,
}

impl<K: DenseKey, V> Default for DenseMap<K, V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<K: DenseKey, V> DenseMap<K, V> {
    /// Constructs a new, empty `IndexVec<I, T>`.
    #[inline]
    pub fn new() -> Self {
        Self::from_raw(Vec::new())
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::from_raw(Vec::with_capacity(capacity))
    }

    #[inline]
    pub fn from_elem_n(elem: V, len: usize) -> Self
    where
        V: Clone,
    {
        Self::from_raw(vec![elem; len])
    }

    #[inline]
    fn from_raw(raw: Vec<V>) -> Self {
        Self { raw, _marker: PhantomData }
    }

    #[inline]
    pub fn get(&self, key: K) -> Option<&V> {
        self.raw.get(key.index())
    }

    #[inline]
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.raw.get_mut(key.index())
    }

    #[inline]
    pub fn replace(&mut self, key: K, value: V) -> V {
        assert!(key.index() < self.raw.len());
        std::mem::replace(&mut self.raw[key.index()], value)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    #[inline]
    pub fn push(&mut self, value: V) -> K {
        let idx = K::new(self.raw.len());
        self.raw.push(value);
        idx
    }

    #[inline]
    #[rustc_lint_query_instability]
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> + DoubleEndedIterator {
        self.raw.iter()
    }

    #[inline]
    #[rustc_lint_query_instability]
    pub fn keys(&self) -> impl ExactSizeIterator<Item = K> {
        (0..self.len()).map(|i| K::new(i))
    }

    #[inline]
    #[rustc_lint_query_instability]
    pub fn iter_enumerated(&self) -> impl ExactSizeIterator<Item = (K, &V)> + DoubleEndedIterator {
        #[allow(rustc::potential_query_instability)]
        (0..self.len()).map(|i| K::new(i)).zip(self.values())
    }
}

impl<'a, K: DenseKey, V> IntoIterator for &'a DenseMap<K, V> {
    type Item = &'a V;
    type IntoIter = Iter<'a, V>;

    #[inline]
    #[rustc_lint_query_instability]
    fn into_iter(self) -> Iter<'a, V> {
        self.raw.iter()
    }
}

impl<K: DenseKey, V> IntoIterator for DenseMap<K, V> {
    type Item = V;
    type IntoIter = IntoIter<V>;

    #[inline]
    #[rustc_lint_query_instability]
    fn into_iter(self) -> IntoIter<V> {
        self.raw.into_iter()
    }
}

impl<K: DenseKey, V: fmt::Debug> fmt::Debug for DenseMap<K, V> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.raw, fmt)
    }
}

impl<K: DenseKey, V> Index<K> for DenseMap<K, V> {
    type Output = V;

    #[inline]
    #[track_caller]
    fn index(&self, index: K) -> &V {
        &self.raw[index.index()]
    }
}

impl<K: DenseKey, V> IndexMut<K> for DenseMap<K, V> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, index: K) -> &mut V {
        &mut self.raw[index.index()]
    }
}

pub struct DenseMapRef<'a, K: DenseKey, V> {
    _marker: PhantomData<K>,
    raw: &'a [V],
}

impl<K: DenseKey, V> Clone for DenseMapRef<'_, K, V> {
    fn clone(&self) -> Self {
        Self::from_raw(self.raw)
    }
}

impl<K: DenseKey, V> Copy for DenseMapRef<'_, K, V> {}

impl<'a, K: DenseKey, V> DenseMapRef<'a, K, V> {
    #[inline]
    pub const fn empty() -> Self {
        Self::from_raw(&[])
    }

    #[inline]
    pub const fn from_raw(raw: &'a [V]) -> Self {
        Self { _marker: PhantomData, raw }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.raw.len()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    #[inline]
    pub fn get(&self, key: K) -> Option<&'a V> {
        self.raw.get(key.index())
    }
}

impl<K: DenseKey, V: fmt::Debug> fmt::Debug for DenseMapRef<'_, K, V> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.raw, fmt)
    }
}

impl<'a, K: DenseKey, V> Index<K> for DenseMapRef<'a, K, V> {
    type Output = V;

    #[inline]
    #[track_caller]
    fn index(&self, index: K) -> &'a V {
        &self.raw[index.index()]
    }
}
