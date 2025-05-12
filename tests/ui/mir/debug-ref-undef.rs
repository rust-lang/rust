//@ run-pass
//@ compile-flags: -g -O -Zmir-opt-level=0 -Zinline-mir=y -Zmir-enable-passes=+ReferencePropagation

#![allow(dead_code)]

use std::marker::PhantomData;

struct RawTable<T> {
    marker: PhantomData<T>,
}

impl<T> RawTable<T> {
    fn iter(&self) -> RawIter<T> {
        RawIter { marker: PhantomData }
    }
}

struct RawIter<T> {
    marker: PhantomData<T>,
}

impl<T> Iterator for RawIter<T> {
    type Item = ();
    fn next(&mut self) -> Option<()> {
        None
    }
}

struct HashMap<T> {
    table: RawTable<T>,
}

struct Iter<T> {
    inner: RawIter<T>, // Removing this breaks the reproducer
}

impl<T> IntoIterator for &HashMap<T> {
    type Item = T;
    type IntoIter = Iter<T>;
    fn into_iter(self) -> Iter<T> {
        Iter { inner: self.table.iter() }
    }
}

impl<T> Iterator for Iter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        None
    }
}

pub fn main() {
    let maybe_hash_set: Option<HashMap<()>> = None;
    for _ in maybe_hash_set.as_ref().unwrap_or(&HashMap { table: RawTable { marker: PhantomData } })
    {
    }
}
