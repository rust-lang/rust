use std::any::{Any, TypeId};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::cmp::{Ord, Ordering, PartialOrd};
use std::collections::HashMap;
use std::convert::AsRef;
use std::ffi::OsStr;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

// FIXME: replace with std::lazy after it gets stabilized and reaches beta
use once_cell::sync::Lazy;

use crate::builder::Step;

pub struct Interned<T>(usize, PhantomData<*const T>);

impl Default for Interned<String> {
    fn default() -> Self {
        INTERNER.intern_string(String::default())
    }
}

impl Default for Interned<PathBuf> {
    fn default() -> Self {
        INTERNER.intern_path(PathBuf::default())
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

impl PartialEq<str> for Interned<String> {
    fn eq(&self, other: &str) -> bool {
        *self == other
    }
}
impl<'a> PartialEq<&'a str> for Interned<String> {
    fn eq(&self, other: &&str) -> bool {
        **self == **other
    }
}
impl<'a, T> PartialEq<&'a Interned<T>> for Interned<T> {
    fn eq(&self, other: &&Self) -> bool {
        self.0 == other.0
    }
}
impl<'a, T> PartialEq<Interned<T>> for &'a Interned<T> {
    fn eq(&self, other: &Interned<T>) -> bool {
        self.0 == other.0
    }
}

unsafe impl<T> Send for Interned<T> {}
unsafe impl<T> Sync for Interned<T> {}

impl fmt::Display for Interned<String> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s: &str = &*self;
        f.write_str(s)
    }
}

impl fmt::Debug for Interned<String> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s: &str = &*self;
        f.write_fmt(format_args!("{:?}", s))
    }
}
impl fmt::Debug for Interned<PathBuf> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s: &Path = &*self;
        f.write_fmt(format_args!("{:?}", s))
    }
}

impl Hash for Interned<String> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let l = INTERNER.strs.lock().unwrap();
        l.get(*self).hash(state)
    }
}

impl Hash for Interned<PathBuf> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let l = INTERNER.paths.lock().unwrap();
        l.get(*self).hash(state)
    }
}

impl Deref for Interned<String> {
    type Target = str;
    fn deref(&self) -> &'static str {
        let l = INTERNER.strs.lock().unwrap();
        unsafe { mem::transmute::<&str, &'static str>(l.get(*self)) }
    }
}

impl Deref for Interned<PathBuf> {
    type Target = Path;
    fn deref(&self) -> &'static Path {
        let l = INTERNER.paths.lock().unwrap();
        unsafe { mem::transmute::<&Path, &'static Path>(l.get(*self)) }
    }
}

impl AsRef<Path> for Interned<PathBuf> {
    fn as_ref(&self) -> &'static Path {
        let l = INTERNER.paths.lock().unwrap();
        unsafe { mem::transmute::<&Path, &'static Path>(l.get(*self)) }
    }
}

impl AsRef<Path> for Interned<String> {
    fn as_ref(&self) -> &'static Path {
        let l = INTERNER.strs.lock().unwrap();
        unsafe { mem::transmute::<&Path, &'static Path>(l.get(*self).as_ref()) }
    }
}

impl AsRef<OsStr> for Interned<PathBuf> {
    fn as_ref(&self) -> &'static OsStr {
        let l = INTERNER.paths.lock().unwrap();
        unsafe { mem::transmute::<&OsStr, &'static OsStr>(l.get(*self).as_ref()) }
    }
}

impl AsRef<OsStr> for Interned<String> {
    fn as_ref(&self) -> &'static OsStr {
        let l = INTERNER.strs.lock().unwrap();
        unsafe { mem::transmute::<&OsStr, &'static OsStr>(l.get(*self).as_ref()) }
    }
}

impl PartialOrd<Interned<String>> for Interned<String> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let l = INTERNER.strs.lock().unwrap();
        l.get(*self).partial_cmp(l.get(*other))
    }
}

impl Ord for Interned<String> {
    fn cmp(&self, other: &Self) -> Ordering {
        let l = INTERNER.strs.lock().unwrap();
        l.get(*self).cmp(l.get(*other))
    }
}

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
    fn intern_borrow<B>(&mut self, item: &B) -> Interned<T>
    where
        B: Eq + Hash + ToOwned<Owned = T> + ?Sized,
        T: Borrow<B>,
    {
        if let Some(i) = self.set.get(&item) {
            return *i;
        }
        let item = item.to_owned();
        let interned = Interned(self.items.len(), PhantomData::<*const T>);
        self.set.insert(item.clone(), interned);
        self.items.push(item);
        interned
    }

    fn intern(&mut self, item: T) -> Interned<T> {
        if let Some(i) = self.set.get(&item) {
            return *i;
        }
        let interned = Interned(self.items.len(), PhantomData::<*const T>);
        self.set.insert(item.clone(), interned);
        self.items.push(item);
        interned
    }

    fn get(&self, i: Interned<T>) -> &T {
        &self.items[i.0]
    }
}

#[derive(Default)]
pub struct Interner {
    strs: Mutex<TyIntern<String>>,
    paths: Mutex<TyIntern<PathBuf>>,
}

impl Interner {
    pub fn intern_str(&self, s: &str) -> Interned<String> {
        self.strs.lock().unwrap().intern_borrow(s)
    }
    pub fn intern_string(&self, s: String) -> Interned<String> {
        self.strs.lock().unwrap().intern(s)
    }

    pub fn intern_path(&self, s: PathBuf) -> Interned<PathBuf> {
        self.paths.lock().unwrap().intern(s)
    }
}

pub static INTERNER: Lazy<Interner> = Lazy::new(Interner::default);

/// This is essentially a `HashMap` which allows storing any type in its input and
/// any type in its output. It is a write-once cache; values are never evicted,
/// which means that references to the value can safely be returned from the
/// `get()` method.
#[derive(Debug)]
pub struct Cache(
    RefCell<
        HashMap<
            TypeId,
            Box<dyn Any>, // actually a HashMap<Step, Interned<Step::Output>>
        >,
    >,
);

impl Cache {
    pub fn new() -> Cache {
        Cache(RefCell::new(HashMap::new()))
    }

    pub fn put<S: Step>(&self, step: S, value: S::Output) {
        let mut cache = self.0.borrow_mut();
        let type_id = TypeId::of::<S>();
        let stepcache = cache
            .entry(type_id)
            .or_insert_with(|| Box::new(HashMap::<S, S::Output>::new()))
            .downcast_mut::<HashMap<S, S::Output>>()
            .expect("invalid type mapped");
        assert!(!stepcache.contains_key(&step), "processing {:?} a second time", step);
        stepcache.insert(step, value);
    }

    pub fn get<S: Step>(&self, step: &S) -> Option<S::Output> {
        let mut cache = self.0.borrow_mut();
        let type_id = TypeId::of::<S>();
        let stepcache = cache
            .entry(type_id)
            .or_insert_with(|| Box::new(HashMap::<S, S::Output>::new()))
            .downcast_mut::<HashMap<S, S::Output>>()
            .expect("invalid type mapped");
        stepcache.get(step).cloned()
    }
}

#[cfg(test)]
impl Cache {
    pub fn all<S: Ord + Copy + Step>(&mut self) -> Vec<(S, S::Output)> {
        let cache = self.0.get_mut();
        let type_id = TypeId::of::<S>();
        let mut v = cache
            .remove(&type_id)
            .map(|b| b.downcast::<HashMap<S, S::Output>>().expect("correct type"))
            .map(|m| m.into_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        v.sort_by_key(|&(a, _)| a);
        v
    }

    pub fn contains<S: Step>(&self) -> bool {
        self.0.borrow().contains_key(&TypeId::of::<S>())
    }
}
