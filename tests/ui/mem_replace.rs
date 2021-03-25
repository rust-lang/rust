// run-rustfix
#![allow(unused_imports)]
#![warn(
    clippy::all,
    clippy::style,
    clippy::mem_replace_option_with_none,
    clippy::mem_replace_with_default
)]

use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, LinkedList, VecDeque};
use std::mem;

fn replace_option_with_none() {
    let mut an_option = Some(1);
    let _ = mem::replace(&mut an_option, None);
    let an_option = &mut Some(1);
    let _ = mem::replace(an_option, None);
}

fn replace_with_default() {
    let mut s = String::from("foo");
    let _ = std::mem::replace(&mut s, String::default());

    let s = &mut String::from("foo");
    let _ = std::mem::replace(s, String::default());
    let _ = std::mem::replace(s, Default::default());

    let mut v = vec![123];
    let _ = std::mem::replace(&mut v, Vec::default());
    let _ = std::mem::replace(&mut v, Default::default());
    let _ = std::mem::replace(&mut v, Vec::new());
    let _ = std::mem::replace(&mut v, vec![]);

    let mut hash_map: HashMap<i32, i32> = HashMap::new();
    let _ = std::mem::replace(&mut hash_map, HashMap::new());

    let mut btree_map: BTreeMap<i32, i32> = BTreeMap::new();
    let _ = std::mem::replace(&mut btree_map, BTreeMap::new());

    let mut vd: VecDeque<i32> = VecDeque::new();
    let _ = std::mem::replace(&mut vd, VecDeque::new());

    let mut hash_set: HashSet<&str> = HashSet::new();
    let _ = std::mem::replace(&mut hash_set, HashSet::new());

    let mut btree_set: BTreeSet<&str> = BTreeSet::new();
    let _ = std::mem::replace(&mut btree_set, BTreeSet::new());

    let mut list: LinkedList<i32> = LinkedList::new();
    let _ = std::mem::replace(&mut list, LinkedList::new());

    let mut binary_heap: BinaryHeap<i32> = BinaryHeap::new();
    let _ = std::mem::replace(&mut binary_heap, BinaryHeap::new());
}

fn main() {
    replace_option_with_none();
    replace_with_default();
}
