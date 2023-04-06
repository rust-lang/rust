//! This module defines a `DynMap` -- a container for heterogeneous maps.
//!
//! This means that `DynMap` stores a bunch of hash maps inside, and those maps
//! can be of different types.
//!
//! It is used like this:
//!
//! ```
//! // keys define submaps of a `DynMap`
//! const STRING_TO_U32: Key<String, u32> = Key::new();
//! const U32_TO_VEC: Key<u32, Vec<bool>> = Key::new();
//!
//! // Note: concrete type, no type params!
//! let mut map = DynMap::new();
//!
//! // To access a specific map, index the `DynMap` by `Key`:
//! map[STRING_TO_U32].insert("hello".to_string(), 92);
//! let value = map[U32_TO_VEC].get(92);
//! assert!(value.is_none());
//! ```
//!
//! This is a work of fiction. Any similarities to Kotlin's `BindingContext` are
//! a coincidence.
pub mod keys;

use std::{
    hash::Hash,
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use anymap::Map;
use rustc_hash::FxHashMap;

pub struct Key<K, V, P = (K, V)> {
    _phantom: PhantomData<(K, V, P)>,
}

impl<K, V, P> Key<K, V, P> {
    pub(crate) const fn new() -> Key<K, V, P> {
        Key { _phantom: PhantomData }
    }
}

impl<K, V, P> Copy for Key<K, V, P> {}

impl<K, V, P> Clone for Key<K, V, P> {
    fn clone(&self) -> Key<K, V, P> {
        *self
    }
}

pub trait Policy {
    type K;
    type V;

    fn insert(map: &mut DynMap, key: Self::K, value: Self::V);
    fn get<'a>(map: &'a DynMap, key: &Self::K) -> Option<&'a Self::V>;
    fn is_empty(map: &DynMap) -> bool;
}

impl<K: Hash + Eq + 'static, V: 'static> Policy for (K, V) {
    type K = K;
    type V = V;
    fn insert(map: &mut DynMap, key: K, value: V) {
        map.map.entry::<FxHashMap<K, V>>().or_insert_with(Default::default).insert(key, value);
    }
    fn get<'a>(map: &'a DynMap, key: &K) -> Option<&'a V> {
        map.map.get::<FxHashMap<K, V>>()?.get(key)
    }
    fn is_empty(map: &DynMap) -> bool {
        map.map.get::<FxHashMap<K, V>>().map_or(true, |it| it.is_empty())
    }
}

pub struct DynMap {
    pub(crate) map: Map,
}

impl Default for DynMap {
    fn default() -> Self {
        DynMap { map: Map::new() }
    }
}

#[repr(transparent)]
pub struct KeyMap<KEY> {
    map: DynMap,
    _phantom: PhantomData<KEY>,
}

impl<P: Policy> KeyMap<Key<P::K, P::V, P>> {
    pub fn insert(&mut self, key: P::K, value: P::V) {
        P::insert(&mut self.map, key, value)
    }
    pub fn get(&self, key: &P::K) -> Option<&P::V> {
        P::get(&self.map, key)
    }

    pub fn is_empty(&self) -> bool {
        P::is_empty(&self.map)
    }
}

impl<P: Policy> Index<Key<P::K, P::V, P>> for DynMap {
    type Output = KeyMap<Key<P::K, P::V, P>>;
    fn index(&self, _key: Key<P::K, P::V, P>) -> &Self::Output {
        // Safe due to `#[repr(transparent)]`.
        unsafe { std::mem::transmute::<&DynMap, &KeyMap<Key<P::K, P::V, P>>>(self) }
    }
}

impl<P: Policy> IndexMut<Key<P::K, P::V, P>> for DynMap {
    fn index_mut(&mut self, _key: Key<P::K, P::V, P>) -> &mut Self::Output {
        // Safe due to `#[repr(transparent)]`.
        unsafe { std::mem::transmute::<&mut DynMap, &mut KeyMap<Key<P::K, P::V, P>>>(self) }
    }
}
