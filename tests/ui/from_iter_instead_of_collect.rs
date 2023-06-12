//@run-rustfix

#![warn(clippy::from_iter_instead_of_collect)]
#![allow(unused_imports, unused_tuple_struct_fields)]
#![allow(clippy::useless_vec)]

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
    }
}

fn main() {
    let iter_expr = std::iter::repeat(5).take(5);
    let _ = Vec::from_iter(iter_expr);

    let _ = HashMap::<usize, &i8>::from_iter(vec![5, 5, 5, 5].iter().enumerate());

    Vec::from_iter(vec![42u32]);

    let a = vec![0, 1, 2];
    assert_eq!(a, Vec::from_iter(0..3));
    assert_eq!(a, Vec::<i32>::from_iter(0..3));

    let mut b = VecDeque::from_iter(0..3);
    b.push_back(4);

    let mut b = VecDeque::<i32>::from_iter(0..3);
    b.push_back(4);

    {
        use std::collections;
        let mut b = collections::VecDeque::<i32>::from_iter(0..3);
        b.push_back(4);
    }

    let values = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')];
    let bm = BTreeMap::from_iter(values.iter().cloned());
    let mut bar = BTreeMap::from_iter(bm.range(0..2));
    bar.insert(&4, &'e');

    let mut bts = BTreeSet::from_iter(0..3);
    bts.insert(2);
    {
        use std::collections;
        let _ = collections::BTreeSet::from_iter(0..3);
        let _ = collections::BTreeSet::<u32>::from_iter(0..3);
    }

    for _i in Vec::from_iter([1, 2, 3].iter()) {}
    for _i in Vec::<&i32>::from_iter([1, 2, 3].iter()) {}
}
