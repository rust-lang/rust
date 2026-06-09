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

fn issue14581() {
    let nums = [0, 1, 2];
    let _ = &String::from_iter(nums.iter().map(|&num| char::from_u32(num).unwrap()));
    //~^ from_iter_instead_of_collect
}

fn test_implicit_generic_args(iter: impl Iterator<Item = &'static i32> + Copy) {
    struct S<'l, T = i32, const A: usize = 3, const B: usize = 3> {
        a: [&'l T; A],
        b: [&'l T; B],
    }

    impl<'l, T, const A: usize, const B: usize> FromIterator<&'l T> for S<'l, T, A, B> {
        fn from_iter<I: IntoIterator<Item = &'l T>>(_: I) -> Self {
            todo!()
        }
    }

    let _ = <S<'static, i32, 7>>::from_iter(iter);
    //~^ from_iter_instead_of_collect

    let _ = <S<'static, i32>>::from_iter(iter);
    //~^ from_iter_instead_of_collect

    let _ = <S<'static, _, 7>>::from_iter(iter);
    //~^ from_iter_instead_of_collect

    let _ = <S<'static, _, 7, 8>>::from_iter(iter);
    //~^ from_iter_instead_of_collect

    let _ = <S<'_, _, 7, 8>>::from_iter(iter);
    //~^ from_iter_instead_of_collect

    let _ = <S<i32>>::from_iter(iter);
    //~^ from_iter_instead_of_collect

    let _ = <S<'_, i32>>::from_iter(iter);
    //~^ from_iter_instead_of_collect

    let _ = <S>::from_iter(iter);
    //~^ from_iter_instead_of_collect
}
