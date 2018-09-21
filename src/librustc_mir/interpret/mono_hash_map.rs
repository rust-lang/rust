//! This is a "monotonic HashMap": A HashMap that, when shared, can be pushed to but not
//! otherwise mutated.  We also Box items in the map. This means we can safely provide
//! shared references into existing items in the HashMap, because they will not be dropped
//! (from being removed) or moved (because they are boxed).
//! The API is is completely tailored to what `memory.rs` needs. It is still in
//! a separate file to minimize the amount of code that has to care about the unsafety.

use std::collections::hash_map::Entry;
use std::cell::RefCell;
use std::hash::Hash;
use std::borrow::Borrow;

use rustc_data_structures::fx::FxHashMap;

#[derive(Debug, Clone)]
pub struct MonoHashMap<K: Hash + Eq, V>(RefCell<FxHashMap<K, Box<V>>>);

impl<K: Hash + Eq, V> Default for MonoHashMap<K, V> {
    fn default() -> Self {
        MonoHashMap(RefCell::new(Default::default()))
    }
}

impl<K: Hash + Eq, V> MonoHashMap<K, V> {
    pub fn contains_key<Q: ?Sized + Hash + Eq>(&mut self, k: &Q) -> bool
        where K: Borrow<Q>
    {
        self.0.get_mut().contains_key(k)
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V>
    {
        self.0.get_mut().insert(k, Box::new(v)).map(|x| *x)
    }

    pub fn remove<Q: ?Sized + Hash + Eq>(&mut self, k: &Q) -> Option<V>
        where K: Borrow<Q>
    {
        self.0.get_mut().remove(k).map(|x| *x)
    }

    pub fn entry(&mut self, k: K) -> Entry<K, Box<V>>
    {
        self.0.get_mut().entry(k)
    }

    pub fn filter_map_collect<T>(&self, mut f: impl FnMut(&K, &V) -> Option<T>) -> Vec<T> {
        self.0.borrow()
            .iter()
            .filter_map(move |(k, v)| f(k, &*v))
            .collect()
    }

    /// The most interesting method: Providing a shared ref without
    /// holding the `RefCell` open, and inserting new data if the key
    /// is not used yet.
    /// `vacant` is called if the key is not found in the map;
    /// if it returns a reference, that is used directly, if it
    /// returns owned data, that is put into the map and returned.
    pub fn get_or<E>(
        &self,
        k: K,
        vacant: impl FnOnce() -> Result<V, E>
    ) -> Result<&V, E> {
        let val: *const V = match self.0.borrow_mut().entry(k) {
            Entry::Occupied(entry) => &**entry.get(),
            Entry::Vacant(entry) => &**entry.insert(Box::new(vacant()?)),
        };
        // This is safe because `val` points into a `Box`, that we know will not move and
        // will also not be dropped as long as the shared reference `self` is live.
        unsafe { Ok(&*val) }
    }

    pub fn get<Q: ?Sized + Hash + Eq>(&self, k: &Q) -> Option<&V>
        where K: Borrow<Q>
    {
        let val: *const V = &**self.0.borrow().get(k)?;
        // This is safe because `val` points into a `Box`, that we know will not move and
        // will also not be dropped as long as the shared reference `self` is live.
        unsafe { Some(&*val) }
    }
}
