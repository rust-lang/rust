// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::any::{Any, TypeId};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::AsRef;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Mutex;

use builder::Step;

pub struct Interned<T>(*mut T);

impl<T> Interned<T> {
    fn as_static(&self) -> &'static T {
        unsafe { mem::transmute::<&T, &'static T>(&*self.0) }
    }
}

impl<T> Default for Interned<T>
where
    T: Intern<T> + Default
{
    fn default() -> Interned<T> {
        T::intern(T::default())
    }
}

impl<T> Copy for Interned<T> {}
impl<T> Clone for Interned<T> {
    fn clone(&self) -> Interned<T> {
        *self
    }
}

impl<A, B> PartialEq<A> for Interned<B>
where
    A: ?Sized + PartialEq<B>,
    B: 'static,
{
    fn eq(&self, other: &A) -> bool {
        other.eq(&*self.as_static())
    }
}

impl<B> Eq for Interned<B>
where
    Interned<B>: PartialEq<Interned<B>>,
{}

impl<'a, T> PartialEq<Interned<T>> for &'a Interned<T> {
    fn eq(&self, other: &Interned<T>) -> bool {
        self.0 == other.0
    }
}

unsafe impl<T> Send for Interned<T> {}
unsafe impl<T> Sync for Interned<T> {}

impl<T: 'static + fmt::Display> fmt::Display for Interned<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.as_static(), f)
    }
}

impl<T: 'static + fmt::Debug> fmt::Debug for Interned<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.as_static(), f)
    }
}

impl<T: 'static + Hash> Hash for Interned<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_static().hash(state)
    }
}

impl<T: 'static + Deref> Deref for Interned<T> {
    type Target = <T as Deref>::Target;
    fn deref(&self) -> &'static Self::Target {
        &*self.as_static()
    }
}

impl<R, I> AsRef<R> for Interned<I>
where
    I: AsRef<R> + 'static,
    R: ?Sized,
{
    fn as_ref(&self) -> &'static R {
        self.as_static().as_ref()
    }
}

pub trait Intern<InternAs>: Sized {
    fn intern(self) -> Interned<InternAs>;
}

impl Intern<String> for String {
    fn intern(self) -> Interned<String> {
        INTERNER.place(self)
    }
}

impl Intern<PathBuf> for PathBuf {
    fn intern(self) -> Interned<PathBuf> {
        INTERNER.place(self)
    }
}

impl<'a, B, I> Intern<I> for &'a B
where
    B: Eq + Hash + ToOwned<Owned=I> + ?Sized + 'static,
    I: Borrow<B> + Clone + Hash + Eq + Send + 'static + Intern<I>,
{
    fn intern(self) -> Interned<I> {
        INTERNER.place_borrow(self)
    }
}

struct TyIntern<T: Eq + Hash> {
    set: HashMap<T, Interned<T>>,
}

impl<T: Hash + Clone + Eq> TyIntern<T> {
    fn new() -> TyIntern<T> {
        TyIntern {
            set: HashMap::new(),
        }
    }

    fn place_borrow<B>(&mut self, item: &B) -> Interned<T>
    where
        B: Eq + Hash + ToOwned<Owned=T> + ?Sized,
        T: Borrow<B>,
    {
        if let Some(i) = self.set.get(&item) {
            return *i;
        }
        self.place(item.to_owned())
    }

    fn place(&mut self, item: T) -> Interned<T> {
        if let Some(i) = self.set.get(&item) {
            return *i;
        }
        let ptr = Box::into_raw(Box::new(item.clone()));
        let interned = Interned(ptr);
        self.set.insert(item, interned);
        interned
    }
}

struct Interner {
    generic: Mutex<Vec<Box<Any + Send + 'static>>>,
}

impl Interner {
    fn new() -> Interner {
        Interner {
            generic: Mutex::new(Vec::new()),
        }
    }

    fn place<T: Hash + Eq + Send + Clone + 'static>(&self, i: T) -> Interned<T> {
        let mut l = self.generic.lock().unwrap();
        for x in l.iter_mut() {
            if let Some(ty_interner) = (&mut **x).downcast_mut::<TyIntern<T>>() {
                return ty_interner.place(i);
            }
        }
        let mut ty_interner = TyIntern::new();
        let interned = ty_interner.place(i);
        l.push(Box::new(ty_interner));
        interned
    }

    fn place_borrow<B, I>(&self, i: &B) -> Interned<I>
    where
        B: Eq + Hash + ToOwned<Owned=I> + ?Sized + 'static,
        I: Borrow<B> + Clone + Hash + Eq + Send + 'static,
    {
        let mut l = self.generic.lock().unwrap();
        for x in l.iter_mut() {
            if let Some(ty_interner) = (&mut **x).downcast_mut::<TyIntern<I>>() {
                return ty_interner.place_borrow(i);
            }
        }
        let mut ty_interner = TyIntern::new();
        let interned = ty_interner.place_borrow(i);
        l.push(Box::new(ty_interner));
        interned
    }
}

lazy_static! {
    static ref INTERNER: Interner = Interner::new();
}

/// This is essentially a HashMap which allows storing any type in its input and
/// any type in its output. It is a write-once cache; values are never evicted,
/// which means that references to the value can safely be returned from the
/// get() method.
#[derive(Debug)]
pub struct Cache(
    RefCell<HashMap<
        TypeId,
        Box<Any>, // actually a HashMap<Step, Interned<Step::Output>>
    >>
);

impl Cache {
    pub fn new() -> Cache {
        Cache(RefCell::new(HashMap::new()))
    }

    pub fn put<S: Step>(&self, step: S, value: S::Output) {
        let mut cache = self.0.borrow_mut();
        let type_id = TypeId::of::<S>();
        let stepcache = cache.entry(type_id)
                        .or_insert_with(|| Box::new(HashMap::<S, S::Output>::new()))
                        .downcast_mut::<HashMap<S, S::Output>>()
                        .expect("invalid type mapped");
        assert!(!stepcache.contains_key(&step), "processing {:?} a second time", step);
        stepcache.insert(step, value);
    }

    pub fn get<S: Step>(&self, step: &S) -> Option<S::Output> {
        let mut cache = self.0.borrow_mut();
        let type_id = TypeId::of::<S>();
        let stepcache = cache.entry(type_id)
                        .or_insert_with(|| Box::new(HashMap::<S, S::Output>::new()))
                        .downcast_mut::<HashMap<S, S::Output>>()
                        .expect("invalid type mapped");
        stepcache.get(step).cloned()
    }
}
