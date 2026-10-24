//! Regression tests which fail if some collections' `PartialEq::eq` impls compare
//! elements when the collections have different sizes.
//! This behavior is not guaranteed either way, so regressing these tests is fine
//! if it is done on purpose.
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, LinkedList};

/// This intentionally has a panicking `PartialEq` impl, to test that various
/// collections' `PartialEq` impls don't actually compare elements if their sizes
/// are unequal.
///
/// This is not advisable in normal code.
#[derive(Debug, Clone, Copy, Hash)]
struct Evil;

impl PartialEq for Evil {
    fn eq(&self, _: &Self) -> bool {
        panic!("Evil::eq is evil");
    }
}
impl Eq for Evil {}

impl PartialOrd for Evil {
    fn partial_cmp(&self, _: &Self) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
}

impl Ord for Evil {
    fn cmp(&self, _: &Self) -> Ordering {
        // Constructing a `BTreeSet`/`BTreeMap` uses `cmp` on the elements,
        // but comparing it with with `==` uses `eq` on the elements,
        // so Evil::cmp doesn't need to be evil.
        Ordering::Equal
    }
}

// check Evil works
#[test]
#[should_panic = "Evil::eq is evil"]
fn evil_eq_works() {
    let v1 = vec![Evil];
    let v2 = vec![Evil];

    _ = v1 == v2;
}

// check various containers don't compare if their sizes are different

#[test]
fn vec_evil_eq() {
    let v1 = vec![Evil];
    let v2 = vec![Evil; 2];

    assert_eq!(false, v1 == v2);
}

#[test]
fn hashset_evil_eq() {
    let s1 = HashSet::from([(0, Evil)]);
    let s2 = HashSet::from([(0, Evil), (1, Evil)]);

    assert_eq!(false, s1 == s2);
}

#[test]
fn hashmap_evil_eq() {
    let m1 = HashMap::from([(0, Evil)]);
    let m2 = HashMap::from([(0, Evil), (1, Evil)]);

    assert_eq!(false, m1 == m2);
}

#[test]
fn btreeset_evil_eq() {
    let s1 = BTreeSet::from([(0, Evil)]);
    let s2 = BTreeSet::from([(0, Evil), (1, Evil)]);

    assert_eq!(false, s1 == s2);
}

#[test]
fn btreemap_evil_eq() {
    let m1 = BTreeMap::from([(0, Evil)]);
    let m2 = BTreeMap::from([(0, Evil), (1, Evil)]);

    assert_eq!(false, m1 == m2);
}

#[test]
fn linkedlist_evil_eq() {
    let m1 = LinkedList::from([Evil]);
    let m2 = LinkedList::from([Evil; 2]);

    assert_eq!(false, m1 == m2);
}
