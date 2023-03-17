//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-strict-provenance
#![feature(btree_drain_filter)]
use std::collections::{BTreeMap, BTreeSet};
use std::mem;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum Foo {
    A(&'static str),
    _B,
    _C,
}

// Gather all references from a mutable iterator and make sure Miri notices if
// using them is dangerous.
fn test_all_refs<'a, T: 'a>(dummy: &mut T, iter: impl Iterator<Item = &'a mut T>) {
    // Gather all those references.
    let mut refs: Vec<&mut T> = iter.collect();
    // Use them all. Twice, to be sure we got all interleavings.
    for r in refs.iter_mut() {
        std::mem::swap(dummy, r);
    }
    for r in refs {
        std::mem::swap(dummy, r);
    }
}

pub fn main() {
    let mut b = BTreeSet::new();
    b.insert(Foo::A("\'"));
    b.insert(Foo::A("/="));
    b.insert(Foo::A("#"));
    b.insert(Foo::A("0o"));
    assert!(b.remove(&Foo::A("/=")));
    assert!(!b.remove(&Foo::A("/=")));

    // Also test a lower-alignment type, where the NodeHeader overlaps with
    // the keys.
    let mut b = BTreeSet::new();
    b.insert(1024u16);
    b.insert(7u16);

    let mut b = BTreeMap::new();
    b.insert(format!("bar"), 1024);
    b.insert(format!("baz"), 7);
    for i in 0..60 {
        b.insert(format!("key{}", i), i);
    }
    test_all_refs(&mut 13, b.values_mut());

    // Test forgetting the drain.
    let mut d = b.drain_filter(|_, i| *i < 30);
    d.next().unwrap();
    mem::forget(d);
}
