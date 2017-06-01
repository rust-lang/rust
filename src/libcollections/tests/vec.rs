// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ascii::AsciiExt;
use std::borrow::Cow;
use std::mem::size_of;
use std::panic;
use std::vec::{Drain, IntoIter};

struct DropCounter<'a> {
    count: &'a mut u32,
}

impl<'a> Drop for DropCounter<'a> {
    fn drop(&mut self) {
        *self.count += 1;
    }
}

#[test]
fn test_small_vec_struct() {
    assert!(size_of::<Vec<u8>>() == size_of::<usize>() * 3);
}

#[test]
fn test_double_drop() {
    struct TwoVec<T> {
        x: Vec<T>,
        y: Vec<T>,
    }

    let (mut count_x, mut count_y) = (0, 0);
    {
        let mut tv = TwoVec {
            x: Vec::new(),
            y: Vec::new(),
        };
        tv.x.push(DropCounter { count: &mut count_x });
        tv.y.push(DropCounter { count: &mut count_y });

        // If Vec had a drop flag, here is where it would be zeroed.
        // Instead, it should rely on its internal state to prevent
        // doing anything significant when dropped multiple times.
        drop(tv.x);

        // Here tv goes out of scope, tv.y should be dropped, but not tv.x.
    }

    assert_eq!(count_x, 1);
    assert_eq!(count_y, 1);
}

#[test]
fn test_reserve() {
    let mut v = Vec::new();
    assert_eq!(v.capacity(), 0);

    v.reserve(2);
    assert!(v.capacity() >= 2);

    for i in 0..16 {
        v.push(i);
    }

    assert!(v.capacity() >= 16);
    v.reserve(16);
    assert!(v.capacity() >= 32);

    v.push(16);

    v.reserve(16);
    assert!(v.capacity() >= 33)
}

#[test]
fn test_extend() {
    let mut v = Vec::new();
    let mut w = Vec::new();

    v.extend(w.clone());
    assert_eq!(v, &[]);

    v.extend(0..3);
    for i in 0..3 {
        w.push(i)
    }

    assert_eq!(v, w);

    v.extend(3..10);
    for i in 3..10 {
        w.push(i)
    }

    assert_eq!(v, w);

    v.extend(w.clone()); // specializes to `append`
    assert!(v.iter().eq(w.iter().chain(w.iter())));

    // Zero sized types
    #[derive(PartialEq, Debug)]
    struct Foo;

    let mut a = Vec::new();
    let b = vec![Foo, Foo];

    a.extend(b);
    assert_eq!(a, &[Foo, Foo]);

    // Double drop
    let mut count_x = 0;
    {
        let mut x = Vec::new();
        let y = vec![DropCounter { count: &mut count_x }];
        x.extend(y);
    }
    assert_eq!(count_x, 1);
}

#[test]
fn test_extend_ref() {
    let mut v = vec![1, 2];
    v.extend(&[3, 4, 5]);

    assert_eq!(v.len(), 5);
    assert_eq!(v, [1, 2, 3, 4, 5]);

    let w = vec![6, 7];
    v.extend(&w);

    assert_eq!(v.len(), 7);
    assert_eq!(v, [1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn test_slice_from_mut() {
    let mut values = vec![1, 2, 3, 4, 5];
    {
        let slice = &mut values[2..];
        assert!(slice == [3, 4, 5]);
        for p in slice {
            *p += 2;
        }
    }

    assert!(values == [1, 2, 5, 6, 7]);
}

#[test]
fn test_slice_to_mut() {
    let mut values = vec![1, 2, 3, 4, 5];
    {
        let slice = &mut values[..2];
        assert!(slice == [1, 2]);
        for p in slice {
            *p += 1;
        }
    }

    assert!(values == [2, 3, 3, 4, 5]);
}

#[test]
fn test_split_at_mut() {
    let mut values = vec![1, 2, 3, 4, 5];
    {
        let (left, right) = values.split_at_mut(2);
        {
            let left: &[_] = left;
            assert!(&left[..left.len()] == &[1, 2]);
        }
        for p in left {
            *p += 1;
        }

        {
            let right: &[_] = right;
            assert!(&right[..right.len()] == &[3, 4, 5]);
        }
        for p in right {
            *p += 2;
        }
    }

    assert_eq!(values, [2, 3, 5, 6, 7]);
}

#[test]
fn test_clone() {
    let v: Vec<i32> = vec![];
    let w = vec![1, 2, 3];

    assert_eq!(v, v.clone());

    let z = w.clone();
    assert_eq!(w, z);
    // they should be disjoint in memory.
    assert!(w.as_ptr() != z.as_ptr())
}

#[test]
fn test_clone_from() {
    let mut v = vec![];
    let three: Vec<Box<_>> = vec![box 1, box 2, box 3];
    let two: Vec<Box<_>> = vec![box 4, box 5];
    // zero, long
    v.clone_from(&three);
    assert_eq!(v, three);

    // equal
    v.clone_from(&three);
    assert_eq!(v, three);

    // long, short
    v.clone_from(&two);
    assert_eq!(v, two);

    // short, long
    v.clone_from(&three);
    assert_eq!(v, three)
}

#[test]
fn test_retain() {
    let mut vec = vec![1, 2, 3, 4];
    vec.retain(|&x| x % 2 == 0);
    assert_eq!(vec, [2, 4]);
}

#[test]
fn test_dedup() {
    fn case(a: Vec<i32>, b: Vec<i32>) {
        let mut v = a;
        v.dedup();
        assert_eq!(v, b);
    }
    case(vec![], vec![]);
    case(vec![1], vec![1]);
    case(vec![1, 1], vec![1]);
    case(vec![1, 2, 3], vec![1, 2, 3]);
    case(vec![1, 1, 2, 3], vec![1, 2, 3]);
    case(vec![1, 2, 2, 3], vec![1, 2, 3]);
    case(vec![1, 2, 3, 3], vec![1, 2, 3]);
    case(vec![1, 1, 2, 2, 2, 3, 3], vec![1, 2, 3]);
}

#[test]
fn test_dedup_by_key() {
    fn case(a: Vec<i32>, b: Vec<i32>) {
        let mut v = a;
        v.dedup_by_key(|i| *i / 10);
        assert_eq!(v, b);
    }
    case(vec![], vec![]);
    case(vec![10], vec![10]);
    case(vec![10, 11], vec![10]);
    case(vec![10, 20, 30], vec![10, 20, 30]);
    case(vec![10, 11, 20, 30], vec![10, 20, 30]);
    case(vec![10, 20, 21, 30], vec![10, 20, 30]);
    case(vec![10, 20, 30, 31], vec![10, 20, 30]);
    case(vec![10, 11, 20, 21, 22, 30, 31], vec![10, 20, 30]);
}

#[test]
fn test_dedup_by() {
    let mut vec = vec!["foo", "bar", "Bar", "baz", "bar"];
    vec.dedup_by(|a, b| a.eq_ignore_ascii_case(b));

    assert_eq!(vec, ["foo", "bar", "baz", "bar"]);
}

#[test]
fn test_dedup_unique() {
    let mut v0: Vec<Box<_>> = vec![box 1, box 1, box 2, box 3];
    v0.dedup();
    let mut v1: Vec<Box<_>> = vec![box 1, box 2, box 2, box 3];
    v1.dedup();
    let mut v2: Vec<Box<_>> = vec![box 1, box 2, box 3, box 3];
    v2.dedup();
    // If the boxed pointers were leaked or otherwise misused, valgrind
    // and/or rt should raise errors.
}

#[test]
fn zero_sized_values() {
    let mut v = Vec::new();
    assert_eq!(v.len(), 0);
    v.push(());
    assert_eq!(v.len(), 1);
    v.push(());
    assert_eq!(v.len(), 2);
    assert_eq!(v.pop(), Some(()));
    assert_eq!(v.pop(), Some(()));
    assert_eq!(v.pop(), None);

    assert_eq!(v.iter().count(), 0);
    v.push(());
    assert_eq!(v.iter().count(), 1);
    v.push(());
    assert_eq!(v.iter().count(), 2);

    for &() in &v {}

    assert_eq!(v.iter_mut().count(), 2);
    v.push(());
    assert_eq!(v.iter_mut().count(), 3);
    v.push(());
    assert_eq!(v.iter_mut().count(), 4);

    for &mut () in &mut v {}
    unsafe {
        v.set_len(0);
    }
    assert_eq!(v.iter_mut().count(), 0);
}

#[test]
fn test_partition() {
    assert_eq!(vec![].into_iter().partition(|x: &i32| *x < 3),
               (vec![], vec![]));
    assert_eq!(vec![1, 2, 3].into_iter().partition(|x| *x < 4),
               (vec![1, 2, 3], vec![]));
    assert_eq!(vec![1, 2, 3].into_iter().partition(|x| *x < 2),
               (vec![1], vec![2, 3]));
    assert_eq!(vec![1, 2, 3].into_iter().partition(|x| *x < 0),
               (vec![], vec![1, 2, 3]));
}

#[test]
fn test_zip_unzip() {
    let z1 = vec![(1, 4), (2, 5), (3, 6)];

    let (left, right): (Vec<_>, Vec<_>) = z1.iter().cloned().unzip();

    assert_eq!((1, 4), (left[0], right[0]));
    assert_eq!((2, 5), (left[1], right[1]));
    assert_eq!((3, 6), (left[2], right[2]));
}

#[test]
fn test_vec_truncate_drop() {
    static mut DROPS: u32 = 0;
    struct Elem(i32);
    impl Drop for Elem {
        fn drop(&mut self) {
            unsafe {
                DROPS += 1;
            }
        }
    }

    let mut v = vec![Elem(1), Elem(2), Elem(3), Elem(4), Elem(5)];
    assert_eq!(unsafe { DROPS }, 0);
    v.truncate(3);
    assert_eq!(unsafe { DROPS }, 2);
    v.truncate(0);
    assert_eq!(unsafe { DROPS }, 5);
}

#[test]
#[should_panic]
fn test_vec_truncate_fail() {
    struct BadElem(i32);
    impl Drop for BadElem {
        fn drop(&mut self) {
            let BadElem(ref mut x) = *self;
            if *x == 0xbadbeef {
                panic!("BadElem panic: 0xbadbeef")
            }
        }
    }

    let mut v = vec![BadElem(1), BadElem(2), BadElem(0xbadbeef), BadElem(4)];
    v.truncate(0);
}

#[test]
fn test_index() {
    let vec = vec![1, 2, 3];
    assert!(vec[1] == 2);
}

#[test]
#[should_panic]
fn test_index_out_of_bounds() {
    let vec = vec![1, 2, 3];
    let _ = vec[3];
}

#[test]
#[should_panic]
fn test_slice_out_of_bounds_1() {
    let x = vec![1, 2, 3, 4, 5];
    &x[!0..];
}

#[test]
#[should_panic]
fn test_slice_out_of_bounds_2() {
    let x = vec![1, 2, 3, 4, 5];
    &x[..6];
}

#[test]
#[should_panic]
fn test_slice_out_of_bounds_3() {
    let x = vec![1, 2, 3, 4, 5];
    &x[!0..4];
}

#[test]
#[should_panic]
fn test_slice_out_of_bounds_4() {
    let x = vec![1, 2, 3, 4, 5];
    &x[1..6];
}

#[test]
#[should_panic]
fn test_slice_out_of_bounds_5() {
    let x = vec![1, 2, 3, 4, 5];
    &x[3..2];
}

#[test]
#[should_panic]
fn test_swap_remove_empty() {
    let mut vec = Vec::<i32>::new();
    vec.swap_remove(0);
}

#[test]
fn test_move_items() {
    let vec = vec![1, 2, 3];
    let mut vec2 = vec![];
    for i in vec {
        vec2.push(i);
    }
    assert_eq!(vec2, [1, 2, 3]);
}

#[test]
fn test_move_items_reverse() {
    let vec = vec![1, 2, 3];
    let mut vec2 = vec![];
    for i in vec.into_iter().rev() {
        vec2.push(i);
    }
    assert_eq!(vec2, [3, 2, 1]);
}

#[test]
fn test_move_items_zero_sized() {
    let vec = vec![(), (), ()];
    let mut vec2 = vec![];
    for i in vec {
        vec2.push(i);
    }
    assert_eq!(vec2, [(), (), ()]);
}

#[test]
fn test_drain_items() {
    let mut vec = vec![1, 2, 3];
    let mut vec2 = vec![];
    for i in vec.drain(..) {
        vec2.push(i);
    }
    assert_eq!(vec, []);
    assert_eq!(vec2, [1, 2, 3]);
}

#[test]
fn test_drain_items_reverse() {
    let mut vec = vec![1, 2, 3];
    let mut vec2 = vec![];
    for i in vec.drain(..).rev() {
        vec2.push(i);
    }
    assert_eq!(vec, []);
    assert_eq!(vec2, [3, 2, 1]);
}

#[test]
fn test_drain_items_zero_sized() {
    let mut vec = vec![(), (), ()];
    let mut vec2 = vec![];
    for i in vec.drain(..) {
        vec2.push(i);
    }
    assert_eq!(vec, []);
    assert_eq!(vec2, [(), (), ()]);
}

#[test]
#[should_panic]
fn test_drain_out_of_bounds() {
    let mut v = vec![1, 2, 3, 4, 5];
    v.drain(5..6);
}

#[test]
fn test_drain_range() {
    let mut v = vec![1, 2, 3, 4, 5];
    for _ in v.drain(4..) {
    }
    assert_eq!(v, &[1, 2, 3, 4]);

    let mut v: Vec<_> = (1..6).map(|x| x.to_string()).collect();
    for _ in v.drain(1..4) {
    }
    assert_eq!(v, &[1.to_string(), 5.to_string()]);

    let mut v: Vec<_> = (1..6).map(|x| x.to_string()).collect();
    for _ in v.drain(1..4).rev() {
    }
    assert_eq!(v, &[1.to_string(), 5.to_string()]);

    let mut v: Vec<_> = vec![(); 5];
    for _ in v.drain(1..4).rev() {
    }
    assert_eq!(v, &[(), ()]);
}

#[test]
fn test_drain_inclusive_range() {
    let mut v = vec!['a', 'b', 'c', 'd', 'e'];
    for _ in v.drain(1...3) {
    }
    assert_eq!(v, &['a', 'e']);

    let mut v: Vec<_> = (0...5).map(|x| x.to_string()).collect();
    for _ in v.drain(1...5) {
    }
    assert_eq!(v, &["0".to_string()]);

    let mut v: Vec<String> = (0...5).map(|x| x.to_string()).collect();
    for _ in v.drain(0...5) {
    }
    assert_eq!(v, Vec::<String>::new());

    let mut v: Vec<_> = (0...5).map(|x| x.to_string()).collect();
    for _ in v.drain(0...3) {
    }
    assert_eq!(v, &["4".to_string(), "5".to_string()]);

    let mut v: Vec<_> = (0...1).map(|x| x.to_string()).collect();
    for _ in v.drain(...0) {
    }
    assert_eq!(v, &["1".to_string()]);
}

#[test]
fn test_drain_max_vec_size() {
    let mut v = Vec::<()>::with_capacity(usize::max_value());
    unsafe { v.set_len(usize::max_value()); }
    for _ in v.drain(usize::max_value() - 1..) {
    }
    assert_eq!(v.len(), usize::max_value() - 1);

    let mut v = Vec::<()>::with_capacity(usize::max_value());
    unsafe { v.set_len(usize::max_value()); }
    for _ in v.drain(usize::max_value() - 1...usize::max_value() - 1) {
    }
    assert_eq!(v.len(), usize::max_value() - 1);
}

#[test]
#[should_panic]
fn test_drain_inclusive_out_of_bounds() {
    let mut v = vec![1, 2, 3, 4, 5];
    v.drain(5...5);
}

#[test]
fn test_splice() {
    let mut v = vec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    v.splice(2..4, a.iter().cloned());
    assert_eq!(v, &[1, 2, 10, 11, 12, 5]);
    v.splice(1..3, Some(20));
    assert_eq!(v, &[1, 20, 11, 12, 5]);
}

#[test]
fn test_splice_inclusive_range() {
    let mut v = vec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    let t1: Vec<_> = v.splice(2...3, a.iter().cloned()).collect();
    assert_eq!(v, &[1, 2, 10, 11, 12, 5]);
    assert_eq!(t1, &[3, 4]);
    let t2: Vec<_> = v.splice(1...2, Some(20)).collect();
    assert_eq!(v, &[1, 20, 11, 12, 5]);
    assert_eq!(t2, &[2, 10]);
}

#[test]
#[should_panic]
fn test_splice_out_of_bounds() {
    let mut v = vec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    v.splice(5..6, a.iter().cloned());
}

#[test]
#[should_panic]
fn test_splice_inclusive_out_of_bounds() {
    let mut v = vec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    v.splice(5...5, a.iter().cloned());
}

#[test]
fn test_splice_items_zero_sized() {
    let mut vec = vec![(), (), ()];
    let vec2 = vec![];
    let t: Vec<_> = vec.splice(1..2, vec2.iter().cloned()).collect();
    assert_eq!(vec, &[(), ()]);
    assert_eq!(t, &[()]);
}

#[test]
fn test_splice_unbounded() {
    let mut vec = vec![1, 2, 3, 4, 5];
    let t: Vec<_> = vec.splice(.., None).collect();
    assert_eq!(vec, &[]);
    assert_eq!(t, &[1, 2, 3, 4, 5]);
}

#[test]
fn test_splice_forget() {
    let mut v = vec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    ::std::mem::forget(v.splice(2..4, a.iter().cloned()));
    assert_eq!(v, &[1, 2]);
}

#[test]
fn test_into_boxed_slice() {
    let xs = vec![1, 2, 3];
    let ys = xs.into_boxed_slice();
    assert_eq!(&*ys, [1, 2, 3]);
}

#[test]
fn test_append() {
    let mut vec = vec![1, 2, 3];
    let mut vec2 = vec![4, 5, 6];
    vec.append(&mut vec2);
    assert_eq!(vec, [1, 2, 3, 4, 5, 6]);
    assert_eq!(vec2, []);
}

#[test]
fn test_split_off() {
    let mut vec = vec![1, 2, 3, 4, 5, 6];
    let vec2 = vec.split_off(4);
    assert_eq!(vec, [1, 2, 3, 4]);
    assert_eq!(vec2, [5, 6]);
}

#[test]
fn test_into_iter_as_slice() {
    let vec = vec!['a', 'b', 'c'];
    let mut into_iter = vec.into_iter();
    assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    let _ = into_iter.next().unwrap();
    assert_eq!(into_iter.as_slice(), &['b', 'c']);
    let _ = into_iter.next().unwrap();
    let _ = into_iter.next().unwrap();
    assert_eq!(into_iter.as_slice(), &[]);
}

#[test]
fn test_into_iter_as_mut_slice() {
    let vec = vec!['a', 'b', 'c'];
    let mut into_iter = vec.into_iter();
    assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    into_iter.as_mut_slice()[0] = 'x';
    into_iter.as_mut_slice()[1] = 'y';
    assert_eq!(into_iter.next().unwrap(), 'x');
    assert_eq!(into_iter.as_slice(), &['y', 'c']);
}

#[test]
fn test_into_iter_debug() {
    let vec = vec!['a', 'b', 'c'];
    let into_iter = vec.into_iter();
    let debug = format!("{:?}", into_iter);
    assert_eq!(debug, "IntoIter(['a', 'b', 'c'])");
}

#[test]
fn test_into_iter_count() {
    assert_eq!(vec![1, 2, 3].into_iter().count(), 3);
}

#[test]
fn test_into_iter_clone() {
    fn iter_equal<I: Iterator<Item = i32>>(it: I, slice: &[i32]) {
        let v: Vec<i32> = it.collect();
        assert_eq!(&v[..], slice);
    }
    let mut it = vec![1, 2, 3].into_iter();
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
fn test_cow_from() {
    let borrowed: &[_] = &["borrowed", "(slice)"];
    let owned = vec!["owned", "(vec)"];
    match (Cow::from(owned.clone()), Cow::from(borrowed)) {
        (Cow::Owned(o), Cow::Borrowed(b)) => assert!(o == owned && b == borrowed),
        _ => panic!("invalid `Cow::from`"),
    }
}

#[test]
fn test_from_cow() {
    let borrowed: &[_] = &["borrowed", "(slice)"];
    let owned = vec!["owned", "(vec)"];
    assert_eq!(Vec::from(Cow::Borrowed(borrowed)), vec!["borrowed", "(slice)"]);
    assert_eq!(Vec::from(Cow::Owned(owned)), vec!["owned", "(vec)"]);
}

#[allow(dead_code)]
fn assert_covariance() {
    fn drain<'new>(d: Drain<'static, &'static str>) -> Drain<'new, &'new str> {
        d
    }
    fn into_iter<'new>(i: IntoIter<&'static str>) -> IntoIter<&'new str> {
        i
    }
}

#[test]
fn test_placement() {
    let mut vec = vec![1];
    assert_eq!(vec.place_back() <- 2, &2);
    assert_eq!(vec.len(), 2);
    assert_eq!(vec.place_back() <- 3, &3);
    assert_eq!(vec.len(), 3);
    assert_eq!(&vec, &[1, 2, 3]);
}

#[test]
fn test_placement_panic() {
    let mut vec = vec![1, 2, 3];
    fn mkpanic() -> usize { panic!() }
    let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| { vec.place_back() <- mkpanic(); }));
    assert_eq!(vec.len(), 3);
}

#[test]
fn from_into_inner() {
    let vec = vec![1, 2, 3];
    let ptr = vec.as_ptr();
    let vec = vec.into_iter().collect::<Vec<_>>();
    assert_eq!(vec, [1, 2, 3]);
    assert_eq!(vec.as_ptr(), ptr);

    let ptr = &vec[1] as *const _;
    let mut it = vec.into_iter();
    it.next().unwrap();
    let vec = it.collect::<Vec<_>>();
    assert_eq!(vec, [2, 3]);
    assert!(ptr != vec.as_ptr());
}

#[test]
fn overaligned_allocations() {
    #[repr(align(256))]
    struct Foo(usize);
    let mut v = vec![Foo(273)];
    for i in 0..0x1000 {
        v.reserve_exact(i);
        assert!(v[0].0 == 273);
        assert!(v.as_ptr() as usize & 0xff == 0);
        v.shrink_to_fit();
        assert!(v[0].0 == 273);
        assert!(v.as_ptr() as usize & 0xff == 0);
    }
}
