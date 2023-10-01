#![warn(clippy::all)]
#![allow(
    clippy::boxed_local,
    clippy::needless_pass_by_value,
    clippy::disallowed_names,
    unused
)]

use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, LinkedList, VecDeque};

macro_rules! boxit {
    ($init:expr, $x:ty) => {
        let _: Box<$x> = Box::new($init);
    };
}

fn test_macro() {
    boxit!(vec![1], Vec<u8>);
}

fn test1(foo: Box<Vec<bool>>) {}
//~^ ERROR: you seem to be trying to use `Box<Vec<..>>`. Consider using just `Vec<..>`

fn test2(foo: Box<dyn Fn(Vec<u32>)>) {
    // pass if #31 is fixed
    foo(vec![1, 2, 3])
}

fn test3(foo: Box<String>) {}
//~^ ERROR: you seem to be trying to use `Box<String>`. Consider using just `String`

fn test4(foo: Box<HashMap<String, String>>) {}
//~^ ERROR: you seem to be trying to use `Box<HashMap<..>>`. Consider using just `HashMap<

fn test5(foo: Box<HashSet<i64>>) {}
//~^ ERROR: you seem to be trying to use `Box<HashSet<..>>`. Consider using just `HashSet<

fn test6(foo: Box<VecDeque<i32>>) {}
//~^ ERROR: you seem to be trying to use `Box<VecDeque<..>>`. Consider using just `VecDequ

fn test7(foo: Box<LinkedList<i16>>) {}
//~^ ERROR: you seem to be trying to use `Box<LinkedList<..>>`. Consider using just `Linke

fn test8(foo: Box<BTreeMap<i8, String>>) {}
//~^ ERROR: you seem to be trying to use `Box<BTreeMap<..>>`. Consider using just `BTreeMa

fn test9(foo: Box<BTreeSet<u64>>) {}
//~^ ERROR: you seem to be trying to use `Box<BTreeSet<..>>`. Consider using just `BTreeSe

fn test10(foo: Box<BinaryHeap<u32>>) {}
//~^ ERROR: you seem to be trying to use `Box<BinaryHeap<..>>`. Consider using just `Binar

fn test_local_not_linted() {
    let _: Box<Vec<bool>>;
}

// All of these test should be allowed because they are part of the
// public api and `avoid_breaking_exported_api` is `false` by default.
pub fn pub_test(foo: Box<Vec<bool>>) {}

pub fn pub_test_ret() -> Box<Vec<bool>> {
    Box::default()
}

fn main() {}
