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
//~^ box_collection

fn test2(foo: Box<dyn Fn(Vec<u32>)>) {
    // pass if #31 is fixed
    foo(vec![1, 2, 3])
}

fn test3(foo: Box<String>) {}
//~^ box_collection

fn test4(foo: Box<HashMap<String, String>>) {}
//~^ box_collection

fn test5(foo: Box<HashSet<i64>>) {}
//~^ box_collection

fn test6(foo: Box<VecDeque<i32>>) {}
//~^ box_collection

fn test7(foo: Box<LinkedList<i16>>) {}
//~^ box_collection

fn test8(foo: Box<BTreeMap<i8, String>>) {}
//~^ box_collection

fn test9(foo: Box<BTreeSet<u64>>) {}
//~^ box_collection

fn test10(foo: Box<BinaryHeap<u32>>) {}
//~^ box_collection

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
