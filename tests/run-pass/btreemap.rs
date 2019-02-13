use std::collections::{BTreeMap, BTreeSet};

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum Foo {
    A(&'static str),
    _B,
    _C,
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
    b.insert(1024);
    b.insert(7);

    let mut b = BTreeMap::new();
    b.insert("bar", 1024);
    b.insert("baz", 7);
    for _val in b.iter_mut() {}
}
