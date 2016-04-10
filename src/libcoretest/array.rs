// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use core::array::FixedSizeArray;

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
fn test_iterator_nth() {
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
fn test_iterator_last() {
    let v = [0, 1, 2, 3, 4];
    assert_eq!(v.into_iter().last().unwrap(), 4);
    assert_eq!([0].into_iter().last().unwrap(), 0);
}

#[test]
fn test_iterator_count() {
    let v = [0, 1, 2, 3, 4];
    assert_eq!(v.clone().into_iter().count(), 5);

    let mut iter2 = v.into_iter();
    iter2.next();
    iter2.next();
    assert_eq!(iter2.count(), 3);
}

#[test]
fn test_iterator_flat_map() {
    assert!((0..5).flat_map(|i| [2 * i, 2 * i + 1]).eq(0..10));
}

#[test]
fn test_iterator_drops() {
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
