// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use core::array::{FixedSizeArray, IntoIter};
use core::convert::TryFrom;

#[test]
fn fixed_size_array() {
    let mut array = [0; 64];
    let mut zero_sized = [(); 64];
    let mut empty_array = [0; 0];
    let mut empty_zero_sized = [(); 0];

    assert_eq!(FixedSizeArray::as_slice(&array).len(), 64);
    assert_eq!(FixedSizeArray::as_slice(&zero_sized).len(), 64);
    assert_eq!(FixedSizeArray::as_slice(&empty_array).len(), 0);
    assert_eq!(FixedSizeArray::as_slice(&empty_zero_sized).len(), 0);

    assert_eq!(FixedSizeArray::as_mut_slice(&mut array).len(), 64);
    assert_eq!(FixedSizeArray::as_mut_slice(&mut zero_sized).len(), 64);
    assert_eq!(FixedSizeArray::as_mut_slice(&mut empty_array).len(), 0);
    assert_eq!(FixedSizeArray::as_mut_slice(&mut empty_zero_sized).len(), 0);
}

#[test]
fn array_try_from() {
    macro_rules! test {
        ($($N:expr)+) => {
            $({
                type Array = [u8; $N];
                let array: Array = [0; $N];
                let slice: &[u8] = &array[..];

                let result = <&Array>::try_from(slice);
                assert_eq!(&array, result.unwrap());
            })+
        }
    }
    test! {
         0  1  2  3  4  5  6  7  8  9
        10 11 12 13 14 15 16 17 18 19
        20 21 22 23 24 25 26 27 28 29
        30 31 32
    }
}

#[test]
fn test_into_iter_as_slice() {
    let array = ['a', 'b', 'c'];
    let mut into_iter = array.into_iter();
    assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    let _ = into_iter.next().unwrap();
    assert_eq!(into_iter.as_slice(), &['b', 'c']);
    let _ = into_iter.next().unwrap();
    let _ = into_iter.next().unwrap();
    assert_eq!(into_iter.as_slice(), &[]);
}

#[test]
fn test_into_iter_as_mut_slice() {
    let array = ['a', 'b', 'c'];
    let mut into_iter = array.into_iter();
    assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    into_iter.as_mut_slice()[0] = 'x';
    into_iter.as_mut_slice()[1] = 'y';
    assert_eq!(into_iter.next().unwrap(), 'x');
    assert_eq!(into_iter.as_slice(), &['y', 'c']);
}

#[test]
fn test_into_iter_debug() {
    let array = ['a', 'b', 'c'];
    let into_iter = array.into_iter();
    let debug = format!("{:?}", into_iter);
    assert_eq!(debug, "IntoIter(['a', 'b', 'c'])");
}

#[test]
fn test_into_iter_clone() {
    fn iter_equal<I: Iterator<Item = i32>>(it: I, slice: &[i32]) {
        let v: Vec<i32> = it.collect();
        assert_eq!(&v[..], slice);
    }
    let mut it = [1, 2, 3].into_iter();
    iter_equal(it.clone(), &[1, 2, 3]);
    assert_eq!(it.next(), Some(1));
    let mut it = it.rev();
    iter_equal(it.clone(), &[3, 2]);
    assert_eq!(it.next(), Some(3));
    iter_equal(it.clone(), &[2]);
    assert_eq!(it.next(), Some(2));
    iter_equal(it.clone(), &[]);
    assert_eq!(it.next(), None);
}

#[test]
fn test_into_iter_nth() {
    let v = [0, 1, 2, 3, 4];
    for i in 0..v.len() {
        assert_eq!(v.clone().into_iter().nth(i).unwrap(), v[i]);
    }
    assert_eq!(v.clone().into_iter().nth(v.len()), None);

    let mut iter = v.into_iter();
    assert_eq!(iter.nth(2).unwrap(), v[2]);
    assert_eq!(iter.nth(1).unwrap(), v[4]);
}

#[test]
fn test_into_iter_last() {
    let v = [0, 1, 2, 3, 4];
    assert_eq!(v.into_iter().last().unwrap(), 4);
    assert_eq!([0].into_iter().last().unwrap(), 0);
}

#[test]
fn test_into_iter_count() {
    let v = [0, 1, 2, 3, 4];
    assert_eq!(v.clone().into_iter().count(), 5);

    let mut iter2 = v.into_iter();
    iter2.next();
    iter2.next();
    assert_eq!(iter2.count(), 3);
}

#[test]
fn test_into_iter_flat_map() {
    assert!((0..5).flat_map(|i| [2 * i, 2 * i + 1]).eq(0..10));
}

#[test]
fn test_into_iter_drops() {
    use core::cell::Cell;

    struct R<'a> {
       i: &'a Cell<usize>,
    }

    impl<'a> Drop for R<'a> {
       fn drop(&mut self) {
            self.i.set(self.i.get() + 1);
        }
    }

    fn r(i: &Cell<usize>) -> R {
        R {
            i: i
        }
    }

    fn v(i: &Cell<usize>) -> [R; 5] {
        [r(i), r(i), r(i), r(i), r(i)]
    }

    let i = Cell::new(0);
    {
        v(&i).into_iter();
    }
    assert_eq!(i.get(), 5);

    let i = Cell::new(0);
    {
        let mut iter = v(&i).into_iter();
        let _x = iter.next();
        assert_eq!(i.get(), 0);
        assert_eq!(iter.count(), 4);
        assert_eq!(i.get(), 4);
    }
    assert_eq!(i.get(), 5);

    let i = Cell::new(0);
    {
        let mut iter = v(&i).into_iter();
        let _x = iter.nth(2);
        assert_eq!(i.get(), 2);
        let _y = iter.last();
        assert_eq!(i.get(), 3);
    }
    assert_eq!(i.get(), 5);

    let i = Cell::new(0);
    for (index, _x) in v(&i).into_iter().enumerate() {
        assert_eq!(i.get(), index);
    }
    assert_eq!(i.get(), 5);

    let i = Cell::new(0);
    for (index, _x) in v(&i).into_iter().rev().enumerate() {
        assert_eq!(i.get(), index);
    }
    assert_eq!(i.get(), 5);
}

#[allow(dead_code)]
fn assert_covariance() {
    fn into_iter<'new>(
        i: IntoIter<&'static str, [&'static str; 10]>,
    ) -> IntoIter<&'new str, [&'new str; 10]> {
        i
    }
}
