//@aux-build:proc_macros.rs
#![warn(clippy::mem_replace_with_default)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, LinkedList, VecDeque};
use std::mem;

fn main() {
    let mut s = String::from("foo");
    let _ = std::mem::replace(&mut s, String::default());
    //~^ mem_replace_with_default
    let _ = std::mem::replace(&mut s, String::new());
    //~^ mem_replace_with_default

    let s = &mut String::from("foo");
    let _ = std::mem::replace(s, String::default());
    //~^ mem_replace_with_default
    let _ = std::mem::replace(s, String::new());
    //~^ mem_replace_with_default
    let _ = std::mem::replace(s, Default::default());
    //~^ mem_replace_with_default

    let mut v = vec![123];
    let _ = std::mem::replace(&mut v, Vec::default());
    //~^ mem_replace_with_default
    let _ = std::mem::replace(&mut v, Default::default());
    //~^ mem_replace_with_default
    let _ = std::mem::replace(&mut v, Vec::new());
    //~^ mem_replace_with_default
    let _ = std::mem::replace(&mut v, vec![]);
    //~^ mem_replace_with_default

    let mut hash_map: HashMap<i32, i32> = HashMap::new();
    let _ = std::mem::replace(&mut hash_map, HashMap::new());
    //~^ mem_replace_with_default

    let mut btree_map: BTreeMap<i32, i32> = BTreeMap::new();
    let _ = std::mem::replace(&mut btree_map, BTreeMap::new());
    //~^ mem_replace_with_default

    let mut vd: VecDeque<i32> = VecDeque::new();
    let _ = std::mem::replace(&mut vd, VecDeque::new());
    //~^ mem_replace_with_default

    let mut hash_set: HashSet<&str> = HashSet::new();
    let _ = std::mem::replace(&mut hash_set, HashSet::new());
    //~^ mem_replace_with_default

    let mut btree_set: BTreeSet<&str> = BTreeSet::new();
    let _ = std::mem::replace(&mut btree_set, BTreeSet::new());
    //~^ mem_replace_with_default

    let mut list: LinkedList<i32> = LinkedList::new();
    let _ = std::mem::replace(&mut list, LinkedList::new());
    //~^ mem_replace_with_default

    let mut binary_heap: BinaryHeap<i32> = BinaryHeap::new();
    let _ = std::mem::replace(&mut binary_heap, BinaryHeap::new());
    //~^ mem_replace_with_default

    let mut tuple = (vec![1, 2], BinaryHeap::<i32>::new());
    let _ = std::mem::replace(&mut tuple, (vec![], BinaryHeap::new()));
    //~^ mem_replace_with_default

    let mut refstr = "hello";
    let _ = std::mem::replace(&mut refstr, "");
    //~^ mem_replace_with_default

    let mut slice: &[i32] = &[1, 2, 3];
    let _ = std::mem::replace(&mut slice, &[]);
    //~^ mem_replace_with_default
}

#[inline_macros]
fn macros(s: &mut String) {
    let _ = inline!(std::mem::replace($s, Default::default()));
    //~^ mem_replace_with_default
    let _ = external!(std::mem::replace($s, Default::default()));
}

// lint is disabled for primitives because in this case `take`
// has no clear benefit over `replace` and sometimes is harder to read
fn dont_lint_primitive() {
    let mut pbool = true;
    let _ = std::mem::replace(&mut pbool, false);

    let mut pint = 5;
    let _ = std::mem::replace(&mut pint, 0);
}

// lint is disabled for expressions that are not used because changing to `take` is not the
// recommended fix. Additionally, the `replace` is #[must_use], so that lint will provide
// the correct suggestion
fn dont_lint_not_used() {
    let mut s = String::from("foo");
    std::mem::replace(&mut s, String::default());
}

#[clippy::msrv = "1.39"]
fn msrv_1_39() {
    let mut s = String::from("foo");
    let _ = std::mem::replace(&mut s, String::default());
}

#[clippy::msrv = "1.40"]
fn msrv_1_40() {
    let mut s = String::from("foo");
    let _ = std::mem::replace(&mut s, String::default());
    //~^ mem_replace_with_default
}

fn issue9824() {
    struct Bar {
        val: String,
    }

    let mut b = Bar {
        val: String::from("bar"),
    };

    let _ = std::mem::replace(&mut b.val, String::default());
    //~^ mem_replace_with_default
}

fn issue15785() {
    let mut text = String::from("foo");
    let replaced = std::mem::replace(dbg!(&mut text), String::default());
    //~^ mem_replace_with_default
}
