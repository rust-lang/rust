//! This module helps you efficiently store and retrieve values using interning.
//!
//! Interning is a neat trick that keeps only one copy of identical values, saving memory
//! and making comparisons super fast. Here, we provide the `Interned<T>` struct and the `Internable` trait
//! to make interning easy for different data types.
//!
//! The `Interner` struct handles caching for common types like `String`, `PathBuf`, and `Vec<String>`,
//! while the `Cache` struct acts as a write-once storage for linking computation steps with their results.
//!
//! # Thread Safety
//!
//! We use `Mutex` to make sure interning and retrieval are thread-safe. But keep in mind—once a value is
//! interned, it sticks around for the entire lifetime of the program.

use std::any::{Any, TypeId};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{LazyLock, Mutex};
use std::{fmt, mem};

use crate::core::builder::Step;

/// Represents an interned value of type `T`, allowing for efficient comparisons and retrieval.
///
/// This struct stores a unique index referencing the interned value within an internal cache.
pub struct Interned<T>(usize, PhantomData<*const T>);

impl<T: Internable + Default> Default for Interned<T> {
    fn default() -> Self {
        T::default().intern()
    }
}

impl<T> Copy for Interned<T> {}
impl<T> Clone for Interned<T> {
    fn clone(&self) -> Interned<T> {
        *self
    }
}

impl<T> PartialEq for Interned<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T> Eq for Interned<T> {}

impl PartialEq<&str> for Interned<String> {
    fn eq(&self, other: &&str) -> bool {
        **self == **other
    }
}

unsafe impl<T> Send for Interned<T> {}
unsafe impl<T> Sync for Interned<T> {}

impl fmt::Display for Interned<String> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s: &str = self;
        f.write_str(s)
    }
}

impl<T, U: ?Sized + fmt::Debug> fmt::Debug for Interned<T>
where
    Self: Deref<Target = U>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s: &U = self;
        f.write_fmt(format_args!("{s:?}"))
    }
}

impl<T: Internable + Hash> Hash for Interned<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let l = T::intern_cache().lock().unwrap();
        l.get(*self).hash(state)
    }
}

impl<T: Internable + Deref> Deref for Interned<T> {
    type Target = T::Target;
    fn deref(&self) -> &Self::Target {
        let l = T::intern_cache().lock().unwrap();
        unsafe { mem::transmute::<&Self::Target, &Self::Target>(l.get(*self)) }
    }
}

impl<T: Internable + AsRef<U>, U: ?Sized> AsRef<U> for Interned<T> {
    fn as_ref(&self) -> &U {
        let l = T::intern_cache().lock().unwrap();
        unsafe { mem::transmute::<&U, &U>(l.get(*self).as_ref()) }
    }
}

impl<T: Internable + PartialOrd> PartialOrd for Interned<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let l = T::intern_cache().lock().unwrap();
        l.get(*self).partial_cmp(l.get(*other))
    }
}

impl<T: Internable + Ord> Ord for Interned<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        let l = T::intern_cache().lock().unwrap();
        l.get(*self).cmp(l.get(*other))
    }
}

/// A structure for managing the interning of values of type `T`.
///
/// `TyIntern<T>` maintains a mapping between values and their interned representations,
/// ensuring that duplicate values are not stored multiple times.
struct TyIntern<T: Clone + Eq> {
    items: Vec<T>,
    set: HashMap<T, Interned<T>>,
}

impl<T: Hash + Clone + Eq> Default for TyIntern<T> {
    fn default() -> Self {
        TyIntern { items: Vec::new(), set: Default::default() }
    }
}

impl<T: Hash + Clone + Eq> TyIntern<T> {
    /// Interns a borrowed value, ensuring it is stored uniquely.
    ///
    /// If the value has been previously interned, the same `Interned<T>` instance is returned.
    fn intern_borrow<B>(&mut self, item: &B) -> Interned<T>
    where
        B: Eq + Hash + ToOwned<Owned = T> + ?Sized,
        T: Borrow<B>,
    {
        if let Some(i) = self.set.get(item) {
            return *i;
        }
        let item = item.to_owned();
        let interned = Interned(self.items.len(), PhantomData::<*const T>);
        self.set.insert(item.clone(), interned);
        self.items.push(item);
        interned
    }

    /// Interns an owned value, storing it uniquely.
    ///
    /// If the value has been previously interned, the existing `Interned<T>` is returned.
    fn intern(&mut self, item: T) -> Interned<T> {
        if let Some(i) = self.set.get(&item) {
            return *i;
        }
        let interned = Interned(self.items.len(), PhantomData::<*const T>);
        self.set.insert(item.clone(), interned);
        self.items.push(item);
        interned
    }

    /// Retrieves a reference to the interned value associated with the given `Interned<T>` instance.
    fn get(&self, i: Interned<T>) -> &T {
        &self.items[i.0]
    }
}

/// A global interner for managing interned values of common types.
///
/// This structure maintains caches for `String`, `PathBuf`, and `Vec<String>`, ensuring efficient storage
/// and retrieval of frequently used values.
#[derive(Default)]
pub struct Interner {
    strs: Mutex<TyIntern<String>>,
}

/// Defines the behavior required for a type to be internable.
///
/// Types implementing this trait must provide access to a static cache and define an `intern` method
/// that ensures values are stored uniquely.
trait Internable: Clone + Eq + Hash + 'static {
    fn intern_cache() -> &'static Mutex<TyIntern<Self>>;

    fn intern(self) -> Interned<Self> {
        Self::intern_cache().lock().unwrap().intern(self)
    }
}

impl Internable for String {
    fn intern_cache() -> &'static Mutex<TyIntern<Self>> {
        &INTERNER.strs
    }
}

impl Interner {
    /// Interns a string reference, ensuring it is stored uniquely.
    ///
    /// If the string has been previously interned, the same `Interned<String>` instance is returned.
    pub fn intern_str(&self, s: &str) -> Interned<String> {
        self.strs.lock().unwrap().intern_borrow(s)
    }
}

/// A global instance of `Interner` that caches common interned values.
pub static INTERNER: LazyLock<Interner> = LazyLock::new(Interner::default);

/// This is essentially a `HashMap` which allows storing any type in its input and
/// any type in its output. It is a write-once cache; values are never evicted,
/// which means that references to the value can safely be returned from the
/// `get()` method.
#[derive(Debug, Default)]
pub struct Cache {
    cache: RefCell<
        HashMap<
            TypeId,
            Box<dyn Any>, // actually a HashMap<Step, Interned<Step::Output>>
        >,
    >,
    #[cfg(test)]
    /// Contains step metadata of executed steps (in the same order in which they were executed).
    /// Useful for tests.
    executed_steps: RefCell<Vec<ExecutedStep>>,
}

#[cfg(test)]
#[derive(Debug)]
pub struct ExecutedStep {
    pub metadata: Option<crate::core::builder::StepMetadata>,
}

impl Cache {
    /// Creates a new empty cache.
    pub fn new() -> Cache {
        Cache::default()
    }

    /// Stores the result of a computation step in the cache.
    pub fn put<S: Step>(&self, step: S, value: S::Output) {
        let mut cache = self.cache.borrow_mut();
        let type_id = TypeId::of::<S>();
        let stepcache = cache
            .entry(type_id)
            .or_insert_with(|| Box::<HashMap<S, S::Output>>::default())
            .downcast_mut::<HashMap<S, S::Output>>()
            .expect("invalid type mapped");
        assert!(!stepcache.contains_key(&step), "processing {step:?} a second time");

        #[cfg(test)]
        {
            let metadata = step.metadata();
            self.executed_steps.borrow_mut().push(ExecutedStep { metadata });
        }

        stepcache.insert(step, value);
    }

    /// Retrieves a cached result for the given step, if available.
    pub fn get<S: Step>(&self, step: &S) -> Option<S::Output> {
        let mut cache = self.cache.borrow_mut();
        let type_id = TypeId::of::<S>();
        let stepcache = cache
            .entry(type_id)
            .or_insert_with(|| Box::<HashMap<S, S::Output>>::default())
            .downcast_mut::<HashMap<S, S::Output>>()
            .expect("invalid type mapped");
        stepcache.get(step).cloned()
    }
}

#[cfg(test)]
impl Cache {
    pub fn all<S: Ord + Step>(&mut self) -> Vec<(S, S::Output)> {
        let cache = self.cache.get_mut();
        let type_id = TypeId::of::<S>();
        let mut v = cache
            .remove(&type_id)
            .map(|b| b.downcast::<HashMap<S, S::Output>>().expect("correct type"))
            .map(|m| m.into_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        v.sort_by_key(|(s, _)| s.clone());
        v
    }

    pub fn contains<S: Step>(&self) -> bool {
        self.cache.borrow().contains_key(&TypeId::of::<S>())
    }

    #[cfg(test)]
    pub fn into_executed_steps(mut self) -> Vec<ExecutedStep> {
        mem::take(&mut self.executed_steps.borrow_mut())
    }
}

#[cfg(test)]
mod tests;
