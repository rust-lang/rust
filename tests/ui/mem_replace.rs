#![allow(unused)]
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

    let mut tuple = (vec![1, 2], BinaryHeap::<i32>::new());
    let _ = std::mem::replace(&mut tuple, (vec![], BinaryHeap::new()));

    let mut refstr = "hello";
    let _ = std::mem::replace(&mut refstr, "");

    let mut slice: &[i32] = &[1, 2, 3];
    let _ = std::mem::replace(&mut slice, &[]);
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

fn main() {
    replace_option_with_none();
    replace_with_default();
    dont_lint_primitive();
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
}

fn issue9824() {
    struct Foo<'a>(Option<&'a str>);
    impl<'a> std::ops::Deref for Foo<'a> {
        type Target = Option<&'a str>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<'a> std::ops::DerefMut for Foo<'a> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    struct Bar {
        opt: Option<u8>,
        val: String,
    }

    let mut f = Foo(Some("foo"));
    let mut b = Bar {
        opt: Some(1),
        val: String::from("bar"),
    };

    // replace option with none
    let _ = std::mem::replace(&mut f.0, None);
    let _ = std::mem::replace(&mut *f, None);
    let _ = std::mem::replace(&mut b.opt, None);
    // replace with default
    let _ = std::mem::replace(&mut b.val, String::default());
}
