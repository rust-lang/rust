#![warn(clippy::from_iter_instead_of_collect)]
#![allow(unused_imports)]
#![allow(clippy::useless_vec, clippy::manual_repeat_n)]

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

struct Foo(Vec<bool>);

impl FromIterator<bool> for Foo {
    fn from_iter<T: IntoIterator<Item = bool>>(_: T) -> Self {
        todo!()
    }
}

impl<'a> FromIterator<&'a bool> for Foo {
    fn from_iter<T: IntoIterator<Item = &'a bool>>(iter: T) -> Self {
        <Self as FromIterator<bool>>::from_iter(iter.into_iter().copied())
        //~^ from_iter_instead_of_collect
    }
}

fn main() {
    let iter_expr = std::iter::repeat(5).take(5);
    let _ = Vec::from_iter(iter_expr);
    //~^ from_iter_instead_of_collect

    let _ = HashMap::<usize, &i8>::from_iter(vec![5, 5, 5, 5].iter().enumerate());
    //~^ from_iter_instead_of_collect

    Vec::from_iter(vec![42u32]);

    let a = vec![0, 1, 2];
    assert_eq!(a, Vec::from_iter(0..3));
    //~^ from_iter_instead_of_collect
    assert_eq!(a, Vec::<i32>::from_iter(0..3));
    //~^ from_iter_instead_of_collect

    let mut b = VecDeque::from_iter(0..3);
    //~^ from_iter_instead_of_collect
    b.push_back(4);

    let mut b = VecDeque::<i32>::from_iter(0..3);
    //~^ from_iter_instead_of_collect
    b.push_back(4);

    {
        use std::collections;
        let mut b = collections::VecDeque::<i32>::from_iter(0..3);
        //~^ from_iter_instead_of_collect
        b.push_back(4);
    }

    let values = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')];
    let bm = BTreeMap::from_iter(values.iter().cloned());
    //~^ from_iter_instead_of_collect
    let mut bar = BTreeMap::from_iter(bm.range(0..2));
    //~^ from_iter_instead_of_collect
    bar.insert(&4, &'e');

    let mut bts = BTreeSet::from_iter(0..3);
    //~^ from_iter_instead_of_collect
    bts.insert(2);
    {
        use std::collections;
        let _ = collections::BTreeSet::from_iter(0..3);
        //~^ from_iter_instead_of_collect
        let _ = collections::BTreeSet::<u32>::from_iter(0..3);
        //~^ from_iter_instead_of_collect
    }

    for _i in Vec::from_iter([1, 2, 3].iter()) {}
    //~^ from_iter_instead_of_collect
    for _i in Vec::<&i32>::from_iter([1, 2, 3].iter()) {}
    //~^ from_iter_instead_of_collect
}
