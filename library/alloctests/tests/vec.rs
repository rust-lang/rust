use core::alloc::{Allocator, Layout};
use core::num::NonZero;
use core::ptr::NonNull;
use core::{assert_eq, assert_ne};
use std::alloc::System;
use std::assert_matches::assert_matches;
use std::borrow::Cow;
use std::cell::Cell;
use std::collections::TryReserveErrorKind::*;
use std::fmt::Debug;
use std::hint;
use std::iter::InPlaceIterable;
use std::mem::swap;
use std::ops::Bound::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::vec::{Drain, IntoIter, PeekMut};

use crate::testing::macros::struct_with_counted_drop;

struct DropCounter<'a> {
    count: &'a mut u32,
}

impl Drop for DropCounter<'_> {
    fn drop(&mut self) {
        *self.count += 1;
    }
}

#[test]
fn test_small_vec_struct() {
    assert_eq!(size_of::<Vec<u8>>(), size_of::<usize>() * 3);
}

#[test]
fn test_double_drop() {
    struct TwoVec<T> {
        x: Vec<T>,
        y: Vec<T>,
    }

    let (mut count_x, mut count_y) = (0, 0);
    {
        let mut tv = TwoVec { x: Vec::new(), y: Vec::new() };
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
fn test_zst_capacity() {
    assert_eq!(Vec::<()>::new().capacity(), usize::MAX);
}

#[test]
fn test_indexing() {
    let v: Vec<isize> = vec![10, 20];
    assert_eq!(v[0], 10);
    assert_eq!(v[1], 20);
    let mut x: usize = 0;
    assert_eq!(v[x], 10);
    assert_eq!(v[x + 1], 20);
    x = x + 1;
    assert_eq!(v[x], 20);
    assert_eq!(v[x - 1], 10);
}

#[test]
fn test_debug_fmt() {
    let vec1: Vec<isize> = vec![];
    assert_eq!("[]", format!("{:?}", vec1));

    let vec2 = vec![0, 1];
    assert_eq!("[0, 1]", format!("{:?}", vec2));

    let slice: &[isize] = &[4, 5];
    assert_eq!("[4, 5]", format!("{slice:?}"));
}

#[test]
fn test_push() {
    let mut v = vec![];
    v.push(1);
    assert_eq!(v, [1]);
    v.push(2);
    assert_eq!(v, [1, 2]);
    v.push(3);
    assert_eq!(v, [1, 2, 3]);
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
fn test_extend_from_slice() {
    let a: Vec<isize> = vec![1, 2, 3, 4, 5];
    let b: Vec<isize> = vec![6, 7, 8, 9, 0];

    let mut v: Vec<isize> = a;

    v.extend_from_slice(&b);

    assert_eq!(v, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]);
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
fn test_slice_from_ref() {
    let values = vec![1, 2, 3, 4, 5];
    let slice = &values[1..3];

    assert_eq!(slice, [2, 3]);
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
    let three: Vec<Box<_>> = vec![Box::new(1), Box::new(2), Box::new(3)];
    let two: Vec<Box<_>> = vec![Box::new(4), Box::new(5)];
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
fn test_retain_predicate_order() {
    for to_keep in [true, false] {
        let mut number_of_executions = 0;
        let mut vec = vec![1, 2, 3, 4];
        let mut next_expected = 1;
        vec.retain(|&x| {
            assert_eq!(next_expected, x);
            next_expected += 1;
            number_of_executions += 1;
            to_keep
        });
        assert_eq!(number_of_executions, 4);
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_retain_pred_panic_with_hole() {
    let v = (0..5).map(Rc::new).collect::<Vec<_>>();
    catch_unwind(AssertUnwindSafe(|| {
        let mut v = v.clone();
        v.retain(|r| match **r {
            0 => true,
            1 => false,
            2 => true,
            _ => panic!(),
        });
    }))
    .unwrap_err();
    // Everything is dropped when predicate panicked.
    assert!(v.iter().all(|r| Rc::strong_count(r) == 1));
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_retain_pred_panic_no_hole() {
    let v = (0..5).map(Rc::new).collect::<Vec<_>>();
    catch_unwind(AssertUnwindSafe(|| {
        let mut v = v.clone();
        v.retain(|r| match **r {
            0 | 1 | 2 => true,
            _ => panic!(),
        });
    }))
    .unwrap_err();
    // Everything is dropped when predicate panicked.
    assert!(v.iter().all(|r| Rc::strong_count(r) == 1));
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_retain_drop_panic() {
    struct Wrap(Rc<i32>);

    impl Drop for Wrap {
        fn drop(&mut self) {
            if *self.0 == 3 {
                panic!();
            }
        }
    }

    let v = (0..5).map(|x| Rc::new(x)).collect::<Vec<_>>();
    catch_unwind(AssertUnwindSafe(|| {
        let mut v = v.iter().map(|r| Wrap(r.clone())).collect::<Vec<_>>();
        v.retain(|w| match *w.0 {
            0 => true,
            1 => false,
            2 => true,
            3 => false, // Drop panic.
            _ => true,
        });
    }))
    .unwrap_err();
    // Other elements are dropped when `drop` of one element panicked.
    // The panicked wrapper also has its Rc dropped.
    assert!(v.iter().all(|r| Rc::strong_count(r) == 1));
}

#[test]
fn test_retain_maybeuninits() {
    // This test aimed to be run under miri.
    use core::mem::MaybeUninit;
    let mut vec: Vec<_> = [1i32, 2, 3, 4].map(|v| MaybeUninit::new(vec![v])).into();
    vec.retain(|x| {
        // SAFETY: Retain must visit every element of Vec in original order and exactly once.
        // Our values is initialized at creation of Vec.
        let v = unsafe { x.assume_init_ref()[0] };
        if v & 1 == 0 {
            return true;
        }
        // SAFETY: Value is initialized.
        // Value wouldn't be dropped by `Vec::retain`
        // because `MaybeUninit` doesn't drop content.
        drop(unsafe { x.assume_init_read() });
        false
    });
    let vec: Vec<i32> = vec
        .into_iter()
        .map(|x| unsafe {
            // SAFETY: All values dropped in retain predicate must be removed by `Vec::retain`.
            // Remaining values are initialized.
            x.assume_init()[0]
        })
        .collect();
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

    let mut vec = vec![("foo", 1), ("foo", 2), ("bar", 3), ("bar", 4), ("bar", 5)];
    vec.dedup_by(|a, b| {
        a.0 == b.0 && {
            b.1 += a.1;
            true
        }
    });

    assert_eq!(vec, [("foo", 3), ("bar", 12)]);
}

#[test]
fn test_dedup_unique() {
    let mut v0: Vec<Box<_>> = vec![Box::new(1), Box::new(1), Box::new(2), Box::new(3)];
    v0.dedup();
    let mut v1: Vec<Box<_>> = vec![Box::new(1), Box::new(2), Box::new(2), Box::new(3)];
    v1.dedup();
    let mut v2: Vec<Box<_>> = vec![Box::new(1), Box::new(2), Box::new(3), Box::new(3)];
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
    assert_eq!([].into_iter().partition(|x: &i32| *x < 3), (vec![], vec![]));
    assert_eq!([1, 2, 3].into_iter().partition(|x| *x < 4), (vec![1, 2, 3], vec![]));
    assert_eq!([1, 2, 3].into_iter().partition(|x| *x < 2), (vec![1], vec![2, 3]));
    assert_eq!([1, 2, 3].into_iter().partition(|x| *x < 0), (vec![], vec![1, 2, 3]));
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
fn test_cmp() {
    let x: &[isize] = &[1, 2, 3, 4, 5];
    let cmp: &[isize] = &[1, 2, 3, 4, 5];
    assert_eq!(&x[..], cmp);
    let cmp: &[isize] = &[3, 4, 5];
    assert_eq!(&x[2..], cmp);
    let cmp: &[isize] = &[1, 2, 3];
    assert_eq!(&x[..3], cmp);
    let cmp: &[isize] = &[2, 3, 4];
    assert_eq!(&x[1..4], cmp);

    let x: Vec<isize> = vec![1, 2, 3, 4, 5];
    let cmp: &[isize] = &[1, 2, 3, 4, 5];
    assert_eq!(&x[..], cmp);
    let cmp: &[isize] = &[3, 4, 5];
    assert_eq!(&x[2..], cmp);
    let cmp: &[isize] = &[1, 2, 3];
    assert_eq!(&x[..3], cmp);
    let cmp: &[isize] = &[2, 3, 4];
    assert_eq!(&x[1..4], cmp);
}

#[test]
fn test_vec_truncate_drop() {
    struct_with_counted_drop!(Elem(i32), DROPS);

    let mut v = vec![Elem(1), Elem(2), Elem(3), Elem(4), Elem(5)];

    assert_eq!(DROPS.get(), 0);
    v.truncate(3);
    assert_eq!(DROPS.get(), 2);
    v.truncate(0);
    assert_eq!(DROPS.get(), 5);
}

#[test]
#[should_panic]
fn test_vec_truncate_fail() {
    struct BadElem(i32);

    impl Drop for BadElem {
        fn drop(&mut self) {
            if let BadElem(0xbadbeef) = self {
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
    let _ = &x[!0..];
}

#[test]
#[should_panic]
fn test_slice_out_of_bounds_2() {
    let x = vec![1, 2, 3, 4, 5];
    let _ = &x[..6];
}

#[test]
#[should_panic]
fn test_slice_out_of_bounds_3() {
    let x = vec![1, 2, 3, 4, 5];
    let _ = &x[!0..4];
}

#[test]
#[should_panic]
fn test_slice_out_of_bounds_4() {
    let x = vec![1, 2, 3, 4, 5];
    let _ = &x[1..6];
}

#[test]
#[should_panic]
fn test_slice_out_of_bounds_5() {
    let x = vec![1, 2, 3, 4, 5];
    let _ = &x[3..2];
}

#[test]
#[should_panic]
fn test_swap_remove_empty() {
    let mut vec = Vec::<i32>::new();
    vec.swap_remove(0);
}

#[test]
fn test_try_remove() {
    let mut vec = vec![1, 2, 3];
    // We are attempting to remove vec[0] which contains 1
    assert_eq!(vec.try_remove(0), Some(1));
    // Now `vec` looks like: [2, 3]
    // We will now try to remove vec[2] which does not exist
    // This should return `None`
    assert_eq!(vec.try_remove(2), None);

    // We will try the same thing with an empty vector
    let mut v: Vec<u8> = vec![];
    assert!(v.try_remove(0).is_none());
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
fn test_drain_empty_vec() {
    let mut vec: Vec<i32> = vec![];
    let mut vec2: Vec<i32> = vec![];
    for i in vec.drain(..) {
        vec2.push(i);
    }
    assert!(vec.is_empty());
    assert!(vec2.is_empty());
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
    for _ in v.drain(4..) {}
    assert_eq!(v, &[1, 2, 3, 4]);

    let mut v: Vec<_> = (1..6).map(|x| x.to_string()).collect();
    for _ in v.drain(1..4) {}
    assert_eq!(v, &[1.to_string(), 5.to_string()]);

    let mut v: Vec<_> = (1..6).map(|x| x.to_string()).collect();
    for _ in v.drain(1..4).rev() {}
    assert_eq!(v, &[1.to_string(), 5.to_string()]);

    let mut v: Vec<_> = vec![(); 5];
    for _ in v.drain(1..4).rev() {}
    assert_eq!(v, &[(), ()]);
}

#[test]
fn test_drain_inclusive_range() {
    let mut v = vec!['a', 'b', 'c', 'd', 'e'];
    for _ in v.drain(1..=3) {}
    assert_eq!(v, &['a', 'e']);

    let mut v: Vec<_> = (0..=5).map(|x| x.to_string()).collect();
    for _ in v.drain(1..=5) {}
    assert_eq!(v, &["0".to_string()]);

    let mut v: Vec<String> = (0..=5).map(|x| x.to_string()).collect();
    for _ in v.drain(0..=5) {}
    assert_eq!(v, Vec::<String>::new());

    let mut v: Vec<_> = (0..=5).map(|x| x.to_string()).collect();
    for _ in v.drain(0..=3) {}
    assert_eq!(v, &["4".to_string(), "5".to_string()]);

    let mut v: Vec<_> = (0..=1).map(|x| x.to_string()).collect();
    for _ in v.drain(..=0) {}
    assert_eq!(v, &["1".to_string()]);
}

#[test]
fn test_drain_max_vec_size() {
    let mut v = Vec::<()>::with_capacity(usize::MAX);
    unsafe {
        v.set_len(usize::MAX);
    }
    for _ in v.drain(usize::MAX - 1..) {}
    assert_eq!(v.len(), usize::MAX - 1);

    let mut v = Vec::<()>::with_capacity(usize::MAX);
    unsafe {
        v.set_len(usize::MAX);
    }
    for _ in v.drain(usize::MAX - 1..=usize::MAX - 1) {}
    assert_eq!(v.len(), usize::MAX - 1);
}

#[test]
#[should_panic]
fn test_drain_index_overflow() {
    let mut v = Vec::<()>::with_capacity(usize::MAX);
    unsafe {
        v.set_len(usize::MAX);
    }
    v.drain(0..=usize::MAX);
}

#[test]
#[should_panic]
fn test_drain_inclusive_out_of_bounds() {
    let mut v = vec![1, 2, 3, 4, 5];
    v.drain(5..=5);
}

#[test]
#[should_panic]
fn test_drain_start_overflow() {
    let mut v = vec![1, 2, 3];
    v.drain((Excluded(usize::MAX), Included(0)));
}

#[test]
#[should_panic]
fn test_drain_end_overflow() {
    let mut v = vec![1, 2, 3];
    v.drain((Included(0), Included(usize::MAX)));
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_drain_leak() {
    struct_with_counted_drop!(D(u32, bool), DROPS => |this: &D| if this.1 { panic!("panic in `drop`"); });

    let mut v = vec![
        D(0, false),
        D(1, false),
        D(2, false),
        D(3, false),
        D(4, true),
        D(5, false),
        D(6, false),
    ];

    catch_unwind(AssertUnwindSafe(|| {
        v.drain(2..=5);
    }))
    .ok();

    assert_eq!(DROPS.get(), 4);
    assert_eq!(v, vec![D(0, false), D(1, false), D(6, false),]);
}

#[test]
fn test_drain_keep_rest() {
    let mut v = vec![0, 1, 2, 3, 4, 5, 6];
    let mut drain = v.drain(1..6);
    assert_eq!(drain.next(), Some(1));
    assert_eq!(drain.next_back(), Some(5));
    assert_eq!(drain.next(), Some(2));

    drain.keep_rest();
    assert_eq!(v, &[0, 3, 4, 6]);
}

#[test]
fn test_drain_keep_rest_all() {
    let mut v = vec![0, 1, 2, 3, 4, 5, 6];
    v.drain(1..6).keep_rest();
    assert_eq!(v, &[0, 1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_drain_keep_rest_none() {
    let mut v = vec![0, 1, 2, 3, 4, 5, 6];
    let mut drain = v.drain(1..6);

    drain.by_ref().for_each(drop);

    drain.keep_rest();
    assert_eq!(v, &[0, 6]);
}

#[test]
fn test_splice() {
    let mut v = vec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    v.splice(2..4, a);
    assert_eq!(v, &[1, 2, 10, 11, 12, 5]);
    v.splice(1..3, Some(20));
    assert_eq!(v, &[1, 20, 11, 12, 5]);
}

#[test]
fn test_splice_inclusive_range() {
    let mut v = vec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    let t1: Vec<_> = v.splice(2..=3, a).collect();
    assert_eq!(v, &[1, 2, 10, 11, 12, 5]);
    assert_eq!(t1, &[3, 4]);
    let t2: Vec<_> = v.splice(1..=2, Some(20)).collect();
    assert_eq!(v, &[1, 20, 11, 12, 5]);
    assert_eq!(t2, &[2, 10]);
}

#[test]
#[should_panic]
fn test_splice_out_of_bounds() {
    let mut v = vec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    v.splice(5..6, a);
}

#[test]
#[should_panic]
fn test_splice_inclusive_out_of_bounds() {
    let mut v = vec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    v.splice(5..=5, a);
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
    std::mem::forget(v.splice(2..4, a));
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
    let orig_ptr = vec.as_ptr();
    let orig_capacity = vec.capacity();

    let split_off = vec.split_off(4);
    assert_eq!(vec, [1, 2, 3, 4]);
    assert_eq!(split_off, [5, 6]);
    assert_eq!(vec.capacity(), orig_capacity);
    assert_eq!(vec.as_ptr(), orig_ptr);
}

#[test]
fn test_split_off_take_all() {
    // Allocate enough capacity that we can tell whether the split-off vector's
    // capacity is based on its size, or (incorrectly) on the original capacity.
    let mut vec = Vec::with_capacity(1000);
    vec.extend([1, 2, 3, 4, 5, 6]);
    let orig_ptr = vec.as_ptr();
    let orig_capacity = vec.capacity();

    let split_off = vec.split_off(0);
    assert_eq!(vec, []);
    assert_eq!(split_off, [1, 2, 3, 4, 5, 6]);
    assert_eq!(vec.capacity(), orig_capacity);
    assert_eq!(vec.as_ptr(), orig_ptr);

    // The split-off vector should be newly-allocated, and should not have
    // stolen the original vector's allocation.
    assert!(split_off.capacity() < orig_capacity);
    assert_ne!(split_off.as_ptr(), orig_ptr);
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
    let debug = format!("{into_iter:?}");
    assert_eq!(debug, "IntoIter(['a', 'b', 'c'])");
}

#[test]
fn test_into_iter_count() {
    assert_eq!([1, 2, 3].into_iter().count(), 3);
}

#[test]
fn test_into_iter_next_chunk() {
    let mut iter = b"lorem".to_vec().into_iter();

    assert_eq!(iter.next_chunk().unwrap(), [b'l', b'o']); // N is inferred as 2
    assert_eq!(iter.next_chunk().unwrap(), [b'r', b'e', b'm']); // N is inferred as 3
    assert_eq!(iter.next_chunk::<4>().unwrap_err().as_slice(), &[]); // N is explicitly 4
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
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_into_iter_leak() {
    struct_with_counted_drop!(D(bool), DROPS => |this: &D| if this.0 { panic!("panic in `drop`"); });

    let v = vec![D(false), D(true), D(false)];

    catch_unwind(move || drop(v.into_iter())).ok();

    assert_eq!(DROPS.get(), 3);
}

#[test]
fn test_into_iter_advance_by() {
    let mut i = vec![1, 2, 3, 4, 5].into_iter();
    assert_eq!(i.advance_by(0), Ok(()));
    assert_eq!(i.advance_back_by(0), Ok(()));
    assert_eq!(i.as_slice(), [1, 2, 3, 4, 5]);

    assert_eq!(i.advance_by(1), Ok(()));
    assert_eq!(i.advance_back_by(1), Ok(()));
    assert_eq!(i.as_slice(), [2, 3, 4]);

    assert_eq!(i.advance_back_by(usize::MAX), Err(NonZero::new(usize::MAX - 3).unwrap()));

    assert_eq!(i.advance_by(usize::MAX), Err(NonZero::new(usize::MAX).unwrap()));

    assert_eq!(i.advance_by(0), Ok(()));
    assert_eq!(i.advance_back_by(0), Ok(()));

    assert_eq!(i.len(), 0);
}

#[test]
fn test_into_iter_drop_allocator() {
    struct ReferenceCountedAllocator<'a>(#[allow(dead_code)] DropCounter<'a>);

    unsafe impl Allocator for ReferenceCountedAllocator<'_> {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, core::alloc::AllocError> {
            System.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            // Safety: Invariants passed to caller.
            unsafe { System.deallocate(ptr, layout) }
        }
    }

    let mut drop_count = 0;

    let allocator = ReferenceCountedAllocator(DropCounter { count: &mut drop_count });
    let _ = Vec::<u32, _>::new_in(allocator);
    assert_eq!(drop_count, 1);

    let allocator = ReferenceCountedAllocator(DropCounter { count: &mut drop_count });
    let _ = Vec::<u32, _>::new_in(allocator).into_iter();
    assert_eq!(drop_count, 2);
}

#[test]
fn test_into_iter_zst() {
    #[derive(Debug, Clone)]
    struct AlignedZstWithDrop([u64; 0]);
    impl Drop for AlignedZstWithDrop {
        fn drop(&mut self) {
            let addr = self as *mut _ as usize;
            assert!(hint::black_box(addr) % align_of::<u64>() == 0);
        }
    }

    const C: AlignedZstWithDrop = AlignedZstWithDrop([0u64; 0]);

    for _ in vec![C].into_iter() {}
    for _ in vec![C; 5].into_iter().rev() {}

    let mut it = vec![C, C].into_iter();
    assert_eq!(it.advance_by(1), Ok(()));
    drop(it);

    let mut it = vec![C, C].into_iter();
    it.next_chunk::<1>().unwrap();
    drop(it);

    let mut it = vec![C, C].into_iter();
    it.next_chunk::<4>().unwrap_err();
    drop(it);
}

#[test]
fn test_from_iter_specialization() {
    let src: Vec<usize> = vec![0usize; 1];
    let srcptr = src.as_ptr();
    let sink = src.into_iter().collect::<Vec<_>>();
    let sinkptr = sink.as_ptr();
    assert_eq!(srcptr, sinkptr);
}

#[test]
fn test_from_iter_partially_drained_in_place_specialization() {
    let src: Vec<usize> = vec![0usize; 10];
    let srcptr = src.as_ptr();
    let mut iter = src.into_iter();
    iter.next();
    iter.next();
    let sink = iter.collect::<Vec<_>>();
    let sinkptr = sink.as_ptr();
    assert_eq!(srcptr, sinkptr);
}

#[test]
fn test_from_iter_specialization_with_iterator_adapters() {
    fn assert_in_place_trait<T: InPlaceIterable>(_: &T) {}
    let owned: Vec<usize> = vec![0usize; 256];
    let refd: Vec<&usize> = owned.iter().collect();
    let src: Vec<&&usize> = refd.iter().collect();
    let srcptr = src.as_ptr();
    let iter = src
        .into_iter()
        .copied()
        .cloned()
        .enumerate()
        .map(|i| i.0 + i.1)
        .zip(std::iter::repeat(1usize))
        .map(|(a, b)| a + b)
        .map_while(Option::Some)
        .skip(1)
        .map(|e| if e != usize::MAX { Ok(NonZero::new(e)) } else { Err(()) });
    assert_in_place_trait(&iter);
    let sink = iter.collect::<Result<Vec<_>, _>>().unwrap();
    let sinkptr = sink.as_ptr();
    assert_eq!(srcptr as *const usize, sinkptr as *const usize);
}

#[test]
fn test_in_place_specialization_step_up_down() {
    fn assert_in_place_trait<T: InPlaceIterable>(_: &T) {}

    let src = vec![0u8; 1024];
    let srcptr = src.as_ptr();
    let src_bytes = src.capacity();
    let iter = src.into_iter().array_chunks::<4>();
    assert_in_place_trait(&iter);
    let sink = iter.collect::<Vec<_>>();
    let sinkptr = sink.as_ptr();
    assert_eq!(srcptr.addr(), sinkptr.addr());
    assert_eq!(src_bytes, sink.capacity() * 4);

    let mut src: Vec<u8> = Vec::with_capacity(17);
    let src_bytes = src.capacity();
    src.resize(8, 0u8);
    let sink: Vec<[u8; 4]> = src.into_iter().array_chunks::<4>().collect();
    let sink_bytes = sink.capacity() * 4;
    assert_ne!(src_bytes, sink_bytes);
    assert_eq!(sink.len(), 2);

    let mut src: Vec<[u8; 3]> = Vec::with_capacity(17);
    src.resize(8, [0; 3]);
    let iter = src.into_iter().map(|[a, b, _]| [a, b]);
    assert_in_place_trait(&iter);
    let sink: Vec<[u8; 2]> = iter.collect();
    assert_eq!(sink.len(), 8);
    assert!(sink.capacity() <= 25);
}

#[test]
fn test_from_iter_specialization_head_tail_drop() {
    let drop_count: Vec<_> = (0..=2).map(|_| Rc::new(())).collect();
    let src: Vec<_> = drop_count.iter().cloned().collect();
    let srcptr = src.as_ptr();
    let iter = src.into_iter();
    let sink: Vec<_> = iter.skip(1).take(1).collect();
    let sinkptr = sink.as_ptr();
    assert_eq!(srcptr, sinkptr, "specialization was applied");
    assert_eq!(Rc::strong_count(&drop_count[0]), 1, "front was dropped");
    assert_eq!(Rc::strong_count(&drop_count[1]), 2, "one element was collected");
    assert_eq!(Rc::strong_count(&drop_count[2]), 1, "tail was dropped");
    assert_eq!(sink.len(), 1);
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_from_iter_specialization_panic_during_iteration_drops() {
    let drop_count: Vec<_> = (0..=2).map(|_| Rc::new(())).collect();
    let src: Vec<_> = drop_count.iter().cloned().collect();
    let iter = src.into_iter();

    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = iter
            .enumerate()
            .filter_map(|(i, e)| {
                if i == 1 {
                    std::panic!("aborting iteration");
                }
                Some(e)
            })
            .collect::<Vec<_>>();
    }));

    assert!(
        drop_count.iter().map(Rc::strong_count).all(|count| count == 1),
        "all items were dropped once"
    );
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_from_iter_specialization_panic_during_drop_doesnt_leak() {
    struct_with_counted_drop!(
        Old(usize), DROP_COUNTER_OLD[|this: &Old| this.0, usize] =>
            |this: &Old| {
                if this.0 == 3 { panic!(); } println!("Dropped Old: {}", this.0)
            }
    );
    struct_with_counted_drop!(
        New(usize), DROP_COUNTER_NEW[|this: &New| this.0, usize] =>
            |this: &New| println!("Dropped New: {}", this.0)
    );

    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let v = vec![Old(0), Old(1), Old(2), Old(3), Old(4)];
        let _ = v.into_iter().map(|x| New(x.0)).take(2).collect::<Vec<_>>();
    }));

    DROP_COUNTER_OLD.with_borrow(|c| assert_eq!(c.get(&0), Some(&1)));
    DROP_COUNTER_OLD.with_borrow(|c| assert_eq!(c.get(&1), Some(&1)));
    DROP_COUNTER_OLD.with_borrow(|c| assert_eq!(c.get(&2), Some(&1)));
    DROP_COUNTER_OLD.with_borrow(|c| assert_eq!(c.get(&3), Some(&1)));
    DROP_COUNTER_OLD.with_borrow(|c| assert_eq!(c.get(&4), Some(&1)));

    DROP_COUNTER_NEW.with_borrow(|c| assert_eq!(c.get(&0), Some(&1)));
    DROP_COUNTER_NEW.with_borrow(|c| assert_eq!(c.get(&1), Some(&1)));
}

// regression test for issue #85322. Peekable previously implemented InPlaceIterable,
// but due to an interaction with IntoIter's current Clone implementation it failed to uphold
// the contract.
#[test]
fn test_collect_after_iterator_clone() {
    let v = vec![0; 5];
    let mut i = v.into_iter().map(|i| i + 1).peekable();
    i.peek();
    let v = i.clone().collect::<Vec<_>>();
    assert_eq!(v, [1, 1, 1, 1, 1]);
    assert!(v.len() <= v.capacity());
}

// regression test for #135103, similar to the one above Flatten/FlatMap had an unsound InPlaceIterable
// implementation.
#[test]
fn test_flatten_clone() {
    const S: String = String::new();

    let v = vec![[S, "Hello World!".into()], [S, S]];
    let mut i = v.into_iter().flatten();
    let _ = i.next();
    let result: Vec<String> = i.clone().collect();
    assert_eq!(result, ["Hello World!", "", ""]);
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

#[test]
fn extract_if_empty() {
    let mut vec: Vec<i32> = vec![];

    {
        let mut iter = vec.extract_if(.., |_| true);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }
    assert_eq!(vec.len(), 0);
    assert_eq!(vec, vec![]);
}

#[test]
fn extract_if_zst() {
    let mut vec = vec![(), (), (), (), ()];
    let initial_len = vec.len();
    let mut count = 0;
    {
        let mut iter = vec.extract_if(.., |_| true);
        assert_eq!(iter.size_hint(), (0, Some(initial_len)));
        while let Some(_) = iter.next() {
            count += 1;
            assert_eq!(iter.size_hint(), (0, Some(initial_len - count)));
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    assert_eq!(count, initial_len);
    assert_eq!(vec.len(), 0);
    assert_eq!(vec, vec![]);
}

#[test]
fn extract_if_false() {
    let mut vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let initial_len = vec.len();
    let mut count = 0;
    {
        let mut iter = vec.extract_if(.., |_| false);
        assert_eq!(iter.size_hint(), (0, Some(initial_len)));
        for _ in iter.by_ref() {
            count += 1;
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    assert_eq!(count, 0);
    assert_eq!(vec.len(), initial_len);
    assert_eq!(vec, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}

#[test]
fn extract_if_true() {
    let mut vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let initial_len = vec.len();
    let mut count = 0;
    {
        let mut iter = vec.extract_if(.., |_| true);
        assert_eq!(iter.size_hint(), (0, Some(initial_len)));
        while let Some(_) = iter.next() {
            count += 1;
            assert_eq!(iter.size_hint(), (0, Some(initial_len - count)));
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    assert_eq!(count, initial_len);
    assert_eq!(vec.len(), 0);
    assert_eq!(vec, vec![]);
}

#[test]
fn extract_if_ranges() {
    let mut vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let mut count = 0;
    let it = vec.extract_if(1..=3, |_| {
        count += 1;
        true
    });
    assert_eq!(it.collect::<Vec<_>>(), vec![1, 2, 3]);
    assert_eq!(vec, vec![0, 4, 5, 6, 7, 8, 9, 10]);
    assert_eq!(count, 3);

    let it = vec.extract_if(1..=3, |_| false);
    assert_eq!(it.collect::<Vec<_>>(), vec![]);
    assert_eq!(vec, vec![0, 4, 5, 6, 7, 8, 9, 10]);
}

#[test]
#[should_panic]
fn extract_if_out_of_bounds() {
    let mut vec = vec![0, 1];
    let _ = vec.extract_if(5.., |_| true).for_each(drop);
}

#[test]
fn extract_if_complex() {
    {
        //                [+xxx++++++xxxxx++++x+x++]
        let mut vec = vec![
            1, 2, 4, 6, 7, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 27, 29, 31, 33, 34, 35, 36, 37,
            39,
        ];

        let removed = vec.extract_if(.., |x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 18, 20, 22, 24, 26, 34, 36]);

        assert_eq!(vec.len(), 14);
        assert_eq!(vec, vec![1, 7, 9, 11, 13, 15, 17, 27, 29, 31, 33, 35, 37, 39]);
    }

    {
        //                [xxx++++++xxxxx++++x+x++]
        let mut vec = vec![
            2, 4, 6, 7, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 27, 29, 31, 33, 34, 35, 36, 37, 39,
        ];

        let removed = vec.extract_if(.., |x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 18, 20, 22, 24, 26, 34, 36]);

        assert_eq!(vec.len(), 13);
        assert_eq!(vec, vec![7, 9, 11, 13, 15, 17, 27, 29, 31, 33, 35, 37, 39]);
    }

    {
        //                [xxx++++++xxxxx++++x+x]
        let mut vec =
            vec![2, 4, 6, 7, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 27, 29, 31, 33, 34, 35, 36];

        let removed = vec.extract_if(.., |x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 18, 20, 22, 24, 26, 34, 36]);

        assert_eq!(vec.len(), 11);
        assert_eq!(vec, vec![7, 9, 11, 13, 15, 17, 27, 29, 31, 33, 35]);
    }

    {
        //                [xxxxxxxxxx+++++++++++]
        let mut vec = vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19];

        let removed = vec.extract_if(.., |x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

        assert_eq!(vec.len(), 10);
        assert_eq!(vec, vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19]);
    }

    {
        //                [+++++++++++xxxxxxxxxx]
        let mut vec = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20];

        let removed = vec.extract_if(.., |x| *x % 2 == 0).collect::<Vec<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

        assert_eq!(vec.len(), 10);
        assert_eq!(vec, vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19]);
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn extract_if_consumed_panic() {
    use std::rc::Rc;
    use std::sync::Mutex;

    struct Check {
        index: usize,
        drop_counts: Rc<Mutex<Vec<usize>>>,
    }

    impl Drop for Check {
        fn drop(&mut self) {
            self.drop_counts.lock().unwrap()[self.index] += 1;
            println!("drop: {}", self.index);
        }
    }

    let check_count = 10;
    let drop_counts = Rc::new(Mutex::new(vec![0_usize; check_count]));
    let mut data: Vec<Check> = (0..check_count)
        .map(|index| Check { index, drop_counts: Rc::clone(&drop_counts) })
        .collect();

    let _ = std::panic::catch_unwind(move || {
        let filter = |c: &mut Check| {
            if c.index == 2 {
                panic!("panic at index: {}", c.index);
            }
            // Verify that if the filter could panic again on another element
            // that it would not cause a double panic and all elements of the
            // vec would still be dropped exactly once.
            if c.index == 4 {
                panic!("panic at index: {}", c.index);
            }
            c.index < 6
        };
        let drain = data.extract_if(.., filter);

        // NOTE: The ExtractIf is explicitly consumed
        drain.for_each(drop);
    });

    let drop_counts = drop_counts.lock().unwrap();
    assert_eq!(check_count, drop_counts.len());

    for (index, count) in drop_counts.iter().cloned().enumerate() {
        assert_eq!(1, count, "unexpected drop count at index: {} (count: {})", index, count);
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn extract_if_unconsumed_panic() {
    use std::rc::Rc;
    use std::sync::Mutex;

    struct Check {
        index: usize,
        drop_counts: Rc<Mutex<Vec<usize>>>,
    }

    impl Drop for Check {
        fn drop(&mut self) {
            self.drop_counts.lock().unwrap()[self.index] += 1;
            println!("drop: {}", self.index);
        }
    }

    let check_count = 10;
    let drop_counts = Rc::new(Mutex::new(vec![0_usize; check_count]));
    let mut data: Vec<Check> = (0..check_count)
        .map(|index| Check { index, drop_counts: Rc::clone(&drop_counts) })
        .collect();

    let _ = std::panic::catch_unwind(move || {
        let filter = |c: &mut Check| {
            if c.index == 2 {
                panic!("panic at index: {}", c.index);
            }
            // Verify that if the filter could panic again on another element
            // that it would not cause a double panic and all elements of the
            // vec would still be dropped exactly once.
            if c.index == 4 {
                panic!("panic at index: {}", c.index);
            }
            c.index < 6
        };
        let _drain = data.extract_if(.., filter);

        // NOTE: The ExtractIf is dropped without being consumed
    });

    let drop_counts = drop_counts.lock().unwrap();
    assert_eq!(check_count, drop_counts.len());

    for (index, count) in drop_counts.iter().cloned().enumerate() {
        assert_eq!(1, count, "unexpected drop count at index: {} (count: {})", index, count);
    }
}

#[test]
fn extract_if_unconsumed() {
    let mut vec = vec![1, 2, 3, 4];
    let drain = vec.extract_if(.., |&mut x| x % 2 != 0);
    drop(drain);
    assert_eq!(vec, [1, 2, 3, 4]);
}

#[test]
fn test_reserve_exact() {
    // This is all the same as test_reserve

    let mut v = Vec::new();
    assert_eq!(v.capacity(), 0);

    v.reserve_exact(2);
    assert!(v.capacity() >= 2);

    for i in 0..16 {
        v.push(i);
    }

    assert!(v.capacity() >= 16);
    v.reserve_exact(16);
    assert!(v.capacity() >= 32);

    v.push(16);

    v.reserve_exact(16);
    assert!(v.capacity() >= 33)
}

#[test]
#[cfg_attr(miri, ignore)] // Miri does not support signalling OOM
fn test_try_with_capacity() {
    let mut vec: Vec<u32> = Vec::try_with_capacity(5).unwrap();
    assert_eq!(0, vec.len());
    assert!(vec.capacity() >= 5 && vec.capacity() <= isize::MAX as usize / 4);
    assert!(vec.spare_capacity_mut().len() >= 5);

    assert!(Vec::<u16>::try_with_capacity(isize::MAX as usize + 1).is_err());
}

#[test]
#[cfg_attr(miri, ignore)] // Miri does not support signalling OOM
fn test_try_reserve() {
    // These are the interesting cases:
    // * exactly isize::MAX should never trigger a CapacityOverflow (can be OOM)
    // * > isize::MAX should always fail
    //    * On 16/32-bit should CapacityOverflow
    //    * On 64-bit should OOM
    // * overflow may trigger when adding `len` to `cap` (in number of elements)
    // * overflow may trigger when multiplying `new_cap` by size_of::<T> (to get bytes)

    const MAX_CAP: usize = isize::MAX as usize;
    const MAX_USIZE: usize = usize::MAX;

    {
        // Note: basic stuff is checked by test_reserve
        let mut empty_bytes: Vec<u8> = Vec::new();

        // Check isize::MAX doesn't count as an overflow
        if let Err(CapacityOverflow) = empty_bytes.try_reserve(MAX_CAP).map_err(|e| e.kind()) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        // Play it again, frank! (just to be sure)
        if let Err(CapacityOverflow) = empty_bytes.try_reserve(MAX_CAP).map_err(|e| e.kind()) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }

        // Check isize::MAX + 1 does count as overflow
        assert_matches!(
            empty_bytes.try_reserve(MAX_CAP + 1).map_err(|e| e.kind()),
            Err(CapacityOverflow),
            "isize::MAX + 1 should trigger an overflow!"
        );

        // Check usize::MAX does count as overflow
        assert_matches!(
            empty_bytes.try_reserve(MAX_USIZE).map_err(|e| e.kind()),
            Err(CapacityOverflow),
            "usize::MAX should trigger an overflow!"
        );
    }

    {
        // Same basic idea, but with non-zero len
        let mut ten_bytes: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        if let Err(CapacityOverflow) = ten_bytes.try_reserve(MAX_CAP - 10).map_err(|e| e.kind()) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if let Err(CapacityOverflow) = ten_bytes.try_reserve(MAX_CAP - 10).map_err(|e| e.kind()) {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }

        assert_matches!(
            ten_bytes.try_reserve(MAX_CAP - 9).map_err(|e| e.kind()),
            Err(CapacityOverflow),
            "isize::MAX + 1 should trigger an overflow!"
        );

        // Should always overflow in the add-to-len
        assert_matches!(
            ten_bytes.try_reserve(MAX_USIZE).map_err(|e| e.kind()),
            Err(CapacityOverflow),
            "usize::MAX should trigger an overflow!"
        );
    }

    {
        // Same basic idea, but with interesting type size
        let mut ten_u32s: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        if let Err(CapacityOverflow) = ten_u32s.try_reserve(MAX_CAP / 4 - 10).map_err(|e| e.kind())
        {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if let Err(CapacityOverflow) = ten_u32s.try_reserve(MAX_CAP / 4 - 10).map_err(|e| e.kind())
        {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }

        assert_matches!(
            ten_u32s.try_reserve(MAX_CAP / 4 - 9).map_err(|e| e.kind()),
            Err(CapacityOverflow),
            "isize::MAX + 1 should trigger an overflow!"
        );

        // Should fail in the mul-by-size
        assert_matches!(
            ten_u32s.try_reserve(MAX_USIZE - 20).map_err(|e| e.kind()),
            Err(CapacityOverflow),
            "usize::MAX should trigger an overflow!"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // Miri does not support signalling OOM
fn test_try_reserve_exact() {
    // This is exactly the same as test_try_reserve with the method changed.
    // See that test for comments.

    const MAX_CAP: usize = isize::MAX as usize;
    const MAX_USIZE: usize = usize::MAX;

    {
        let mut empty_bytes: Vec<u8> = Vec::new();

        if let Err(CapacityOverflow) = empty_bytes.try_reserve_exact(MAX_CAP).map_err(|e| e.kind())
        {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if let Err(CapacityOverflow) = empty_bytes.try_reserve_exact(MAX_CAP).map_err(|e| e.kind())
        {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }

        assert_matches!(
            empty_bytes.try_reserve_exact(MAX_CAP + 1).map_err(|e| e.kind()),
            Err(CapacityOverflow),
            "isize::MAX + 1 should trigger an overflow!"
        );

        assert_matches!(
            empty_bytes.try_reserve_exact(MAX_USIZE).map_err(|e| e.kind()),
            Err(CapacityOverflow),
            "usize::MAX should trigger an overflow!"
        );
    }

    {
        let mut ten_bytes: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        if let Err(CapacityOverflow) =
            ten_bytes.try_reserve_exact(MAX_CAP - 10).map_err(|e| e.kind())
        {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if let Err(CapacityOverflow) =
            ten_bytes.try_reserve_exact(MAX_CAP - 10).map_err(|e| e.kind())
        {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }

        assert_matches!(
            ten_bytes.try_reserve_exact(MAX_CAP - 9).map_err(|e| e.kind()),
            Err(CapacityOverflow),
            "isize::MAX + 1 should trigger an overflow!"
        );

        assert_matches!(
            ten_bytes.try_reserve_exact(MAX_USIZE).map_err(|e| e.kind()),
            Err(CapacityOverflow),
            "usize::MAX should trigger an overflow!"
        );
    }

    {
        let mut ten_u32s: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        if let Err(CapacityOverflow) =
            ten_u32s.try_reserve_exact(MAX_CAP / 4 - 10).map_err(|e| e.kind())
        {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }
        if let Err(CapacityOverflow) =
            ten_u32s.try_reserve_exact(MAX_CAP / 4 - 10).map_err(|e| e.kind())
        {
            panic!("isize::MAX shouldn't trigger an overflow!");
        }

        assert_matches!(
            ten_u32s.try_reserve_exact(MAX_CAP / 4 - 9).map_err(|e| e.kind()),
            Err(CapacityOverflow),
            "isize::MAX + 1 should trigger an overflow!"
        );

        assert_matches!(
            ten_u32s.try_reserve_exact(MAX_USIZE - 20).map_err(|e| e.kind()),
            Err(CapacityOverflow),
            "usize::MAX should trigger an overflow!"
        );
    }
}

#[test]
fn test_stable_pointers() {
    /// Pull an element from the iterator, then drop it.
    /// Useful to cover both the `next` and `drop` paths of an iterator.
    fn next_then_drop<I: Iterator>(mut i: I) {
        i.next().unwrap();
        drop(i);
    }

    // Test that, if we reserved enough space, adding and removing elements does not
    // invalidate references into the vector (such as `v0`). This test also
    // runs in Miri, which would detect such problems.
    // Note that this test does *not* constitute a stable guarantee that all these functions do not
    // reallocate! Only what is explicitly documented at
    // <https://doc.rust-lang.org/nightly/std/vec/struct.Vec.html#guarantees> is stably guaranteed.
    let mut v = Vec::with_capacity(128);
    v.push(13);

    // Laundering the lifetime -- we take care that `v` does not reallocate, so that's okay.
    let v0 = &mut v[0];
    let v0 = unsafe { &mut *(v0 as *mut _) };
    // Now do a bunch of things and occasionally use `v0` again to assert it is still valid.

    // Pushing/inserting and popping/removing
    v.push(1);
    v.push(2);
    v.insert(1, 1);
    assert_eq!(*v0, 13);
    v.remove(1);
    v.pop().unwrap();
    assert_eq!(*v0, 13);
    v.push(1);
    v.swap_remove(1);
    assert_eq!(v.len(), 2);
    v.swap_remove(1); // swap_remove the last element
    assert_eq!(*v0, 13);

    // Appending
    v.append(&mut vec![27, 19]);
    assert_eq!(*v0, 13);

    // Extending
    v.extend_from_slice(&[1, 2]);
    v.extend(&[1, 2]); // `slice::Iter` (with `T: Copy`) specialization
    v.extend(vec![2, 3]); // `vec::IntoIter` specialization
    v.extend(std::iter::once(3)); // `TrustedLen` specialization
    v.extend(std::iter::empty::<i32>()); // `TrustedLen` specialization with empty iterator
    v.extend(std::iter::once(3).filter(|_| true)); // base case
    v.extend(std::iter::once(&3)); // `cloned` specialization
    assert_eq!(*v0, 13);

    // Truncation
    v.truncate(2);
    assert_eq!(*v0, 13);

    // Resizing
    v.resize_with(v.len() + 10, || 42);
    assert_eq!(*v0, 13);
    v.resize_with(2, || panic!());
    assert_eq!(*v0, 13);

    // No-op reservation
    v.reserve(32);
    v.reserve_exact(32);
    assert_eq!(*v0, 13);

    // Partial draining
    v.resize_with(10, || 42);
    next_then_drop(v.drain(5..));
    assert_eq!(*v0, 13);

    // Splicing
    v.resize_with(10, || 42);
    next_then_drop(v.splice(5.., vec![1, 2, 3, 4, 5])); // empty tail after range
    assert_eq!(*v0, 13);
    next_then_drop(v.splice(5..8, vec![1])); // replacement is smaller than original range
    assert_eq!(*v0, 13);
    next_then_drop(v.splice(5..6, [1; 10].into_iter().filter(|_| true))); // lower bound not exact
    assert_eq!(*v0, 13);

    // spare_capacity_mut
    v.spare_capacity_mut();
    assert_eq!(*v0, 13);

    // Smoke test that would fire even outside Miri if an actual relocation happened.
    // Also ensures the pointer is still writeable after all this.
    *v0 -= 13;
    assert_eq!(v[0], 0);
}

// https://github.com/rust-lang/rust/pull/49496 introduced specialization based on:
//
// ```
// unsafe impl<T: ?Sized> IsZero for *mut T {
//     fn is_zero(&self) -> bool {
//         (*self).is_null()
//     }
// }
// ```
//
// … to call `RawVec::with_capacity_zeroed` for creating `Vec<*mut T>`,
// which is incorrect for fat pointers since `<*mut T>::is_null` only looks at the data component.
// That is, a fat pointer can be “null” without being made entirely of zero bits.
#[test]
fn vec_macro_repeating_null_raw_fat_pointer() {
    let raw_dyn = &mut (|| ()) as &mut dyn Fn() as *mut dyn Fn();
    let vtable = dbg!(ptr_metadata(raw_dyn));
    let null_raw_dyn = ptr_from_raw_parts(std::ptr::null_mut(), vtable);
    assert!(null_raw_dyn.is_null());

    let vec = vec![null_raw_dyn; 1];
    dbg!(ptr_metadata(vec[0]));
    assert!(std::ptr::eq(vec[0], null_raw_dyn));

    // Polyfill for https://github.com/rust-lang/rfcs/pull/2580

    fn ptr_metadata(ptr: *mut dyn Fn()) -> *mut () {
        unsafe { std::mem::transmute::<*mut dyn Fn(), DynRepr>(ptr).vtable }
    }

    fn ptr_from_raw_parts(data: *mut (), vtable: *mut ()) -> *mut dyn Fn() {
        unsafe { std::mem::transmute::<DynRepr, *mut dyn Fn()>(DynRepr { data, vtable }) }
    }

    #[repr(C)]
    struct DynRepr {
        data: *mut (),
        vtable: *mut (),
    }
}

// This test will likely fail if you change the capacities used in
// `RawVec::grow_amortized`.
#[test]
fn test_push_growth_strategy() {
    // If the element size is 1, we jump from 0 to 8, then double.
    {
        let mut v1: Vec<u8> = vec![];
        assert_eq!(v1.capacity(), 0);

        for _ in 0..8 {
            v1.push(0);
            assert_eq!(v1.capacity(), 8);
        }

        for _ in 8..16 {
            v1.push(0);
            assert_eq!(v1.capacity(), 16);
        }

        for _ in 16..32 {
            v1.push(0);
            assert_eq!(v1.capacity(), 32);
        }

        for _ in 32..64 {
            v1.push(0);
            assert_eq!(v1.capacity(), 64);
        }
    }

    // If the element size is 2..=1024, we jump from 0 to 4, then double.
    {
        let mut v2: Vec<u16> = vec![];
        let mut v1024: Vec<[u8; 1024]> = vec![];
        assert_eq!(v2.capacity(), 0);
        assert_eq!(v1024.capacity(), 0);

        for _ in 0..4 {
            v2.push(0);
            v1024.push([0; 1024]);
            assert_eq!(v2.capacity(), 4);
            assert_eq!(v1024.capacity(), 4);
        }

        for _ in 4..8 {
            v2.push(0);
            v1024.push([0; 1024]);
            assert_eq!(v2.capacity(), 8);
            assert_eq!(v1024.capacity(), 8);
        }

        for _ in 8..16 {
            v2.push(0);
            v1024.push([0; 1024]);
            assert_eq!(v2.capacity(), 16);
            assert_eq!(v1024.capacity(), 16);
        }

        for _ in 16..32 {
            v2.push(0);
            v1024.push([0; 1024]);
            assert_eq!(v2.capacity(), 32);
            assert_eq!(v1024.capacity(), 32);
        }

        for _ in 32..64 {
            v2.push(0);
            v1024.push([0; 1024]);
            assert_eq!(v2.capacity(), 64);
            assert_eq!(v1024.capacity(), 64);
        }
    }

    // If the element size is > 1024, we jump from 0 to 1, then double.
    {
        let mut v1025: Vec<[u8; 1025]> = vec![];
        assert_eq!(v1025.capacity(), 0);

        for _ in 0..1 {
            v1025.push([0; 1025]);
            assert_eq!(v1025.capacity(), 1);
        }

        for _ in 1..2 {
            v1025.push([0; 1025]);
            assert_eq!(v1025.capacity(), 2);
        }

        for _ in 2..4 {
            v1025.push([0; 1025]);
            assert_eq!(v1025.capacity(), 4);
        }

        for _ in 4..8 {
            v1025.push([0; 1025]);
            assert_eq!(v1025.capacity(), 8);
        }

        for _ in 8..16 {
            v1025.push([0; 1025]);
            assert_eq!(v1025.capacity(), 16);
        }

        for _ in 16..32 {
            v1025.push([0; 1025]);
            assert_eq!(v1025.capacity(), 32);
        }

        for _ in 32..64 {
            v1025.push([0; 1025]);
            assert_eq!(v1025.capacity(), 64);
        }
    }
}

macro_rules! generate_assert_eq_vec_and_prim {
    ($name:ident<$B:ident>($type:ty)) => {
        fn $name<A: PartialEq<$B> + Debug, $B: Debug>(a: Vec<A>, b: $type) {
            assert!(a == b);
            assert_eq!(a, b);
        }
    };
}

generate_assert_eq_vec_and_prim! { assert_eq_vec_and_slice  <B>(&[B])   }
generate_assert_eq_vec_and_prim! { assert_eq_vec_and_array_3<B>([B; 3]) }

#[test]
fn partialeq_vec_and_prim() {
    assert_eq_vec_and_slice(vec![1, 2, 3], &[1, 2, 3]);
    assert_eq_vec_and_array_3(vec![1, 2, 3], [1, 2, 3]);
}

macro_rules! assert_partial_eq_valid {
    ($a2:expr, $a3:expr; $b2:expr, $b3: expr) => {
        assert!($a2 == $b2);
        assert!($a2 != $b3);
        assert!($a3 != $b2);
        assert!($a3 == $b3);
        assert_eq!($a2, $b2);
        assert_ne!($a2, $b3);
        assert_ne!($a3, $b2);
        assert_eq!($a3, $b3);
    };
}

#[test]
fn partialeq_vec_full() {
    let vec2: Vec<_> = vec![1, 2];
    let vec3: Vec<_> = vec![1, 2, 3];
    let slice2: &[_] = &[1, 2];
    let slice3: &[_] = &[1, 2, 3];
    let slicemut2: &[_] = &mut [1, 2];
    let slicemut3: &[_] = &mut [1, 2, 3];
    let array2: [_; 2] = [1, 2];
    let array3: [_; 3] = [1, 2, 3];
    let arrayref2: &[_; 2] = &[1, 2];
    let arrayref3: &[_; 3] = &[1, 2, 3];

    assert_partial_eq_valid!(vec2,vec3; vec2,vec3);
    assert_partial_eq_valid!(vec2,vec3; slice2,slice3);
    assert_partial_eq_valid!(vec2,vec3; slicemut2,slicemut3);
    assert_partial_eq_valid!(slice2,slice3; vec2,vec3);
    assert_partial_eq_valid!(slicemut2,slicemut3; vec2,vec3);
    assert_partial_eq_valid!(vec2,vec3; array2,array3);
    assert_partial_eq_valid!(vec2,vec3; arrayref2,arrayref3);
    assert_partial_eq_valid!(vec2,vec3; arrayref2[..],arrayref3[..]);
}

#[test]
fn test_vec_cycle() {
    #[derive(Debug)]
    struct C<'a> {
        v: Vec<Cell<Option<&'a C<'a>>>>,
    }

    impl<'a> C<'a> {
        fn new() -> C<'a> {
            C { v: Vec::new() }
        }
    }

    let mut c1 = C::new();
    let mut c2 = C::new();
    let mut c3 = C::new();

    // Push
    c1.v.push(Cell::new(None));
    c1.v.push(Cell::new(None));

    c2.v.push(Cell::new(None));
    c2.v.push(Cell::new(None));

    c3.v.push(Cell::new(None));
    c3.v.push(Cell::new(None));

    // Set
    c1.v[0].set(Some(&c2));
    c1.v[1].set(Some(&c3));

    c2.v[0].set(Some(&c2));
    c2.v[1].set(Some(&c3));

    c3.v[0].set(Some(&c1));
    c3.v[1].set(Some(&c2));
}

#[test]
fn test_vec_cycle_wrapped() {
    struct Refs<'a> {
        v: Vec<Cell<Option<&'a C<'a>>>>,
    }

    struct C<'a> {
        refs: Refs<'a>,
    }

    impl<'a> Refs<'a> {
        fn new() -> Refs<'a> {
            Refs { v: Vec::new() }
        }
    }

    impl<'a> C<'a> {
        fn new() -> C<'a> {
            C { refs: Refs::new() }
        }
    }

    let mut c1 = C::new();
    let mut c2 = C::new();
    let mut c3 = C::new();

    c1.refs.v.push(Cell::new(None));
    c1.refs.v.push(Cell::new(None));
    c2.refs.v.push(Cell::new(None));
    c2.refs.v.push(Cell::new(None));
    c3.refs.v.push(Cell::new(None));
    c3.refs.v.push(Cell::new(None));

    c1.refs.v[0].set(Some(&c2));
    c1.refs.v[1].set(Some(&c3));
    c2.refs.v[0].set(Some(&c2));
    c2.refs.v[1].set(Some(&c3));
    c3.refs.v[0].set(Some(&c1));
    c3.refs.v[1].set(Some(&c2));
}

#[test]
fn test_zero_sized_capacity() {
    for len in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256] {
        let v = Vec::<()>::with_capacity(len);
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), usize::MAX);
    }
}

#[test]
fn test_zero_sized_vec_push() {
    const N: usize = 8;

    for len in 0..N {
        let mut tester = Vec::with_capacity(len);
        assert_eq!(tester.len(), 0);
        assert!(tester.capacity() >= len);
        for _ in 0..len {
            tester.push(());
        }
        assert_eq!(tester.len(), len);
        assert_eq!(tester.iter().count(), len);
        tester.clear();
    }
}

#[test]
fn test_vec_macro_repeat() {
    assert_eq!(vec![1; 3], vec![1, 1, 1]);
    assert_eq!(vec![1; 2], vec![1, 1]);
    assert_eq!(vec![1; 1], vec![1]);
    assert_eq!(vec![1; 0], vec![]);

    // from_elem syntax (see RFC 832)
    let el = Box::new(1);
    let n = 3;
    assert_eq!(vec![el; n], vec![Box::new(1), Box::new(1), Box::new(1)]);
}

#[test]
fn test_vec_swap() {
    let mut a: Vec<isize> = vec![0, 1, 2, 3, 4, 5, 6];
    a.swap(2, 4);
    assert_eq!(a[2], 4);
    assert_eq!(a[4], 2);
    let mut n = 42;
    swap(&mut n, &mut a[0]);
    assert_eq!(a[0], 42);
    assert_eq!(n, 0);
}

#[test]
fn test_extend_from_within_spec() {
    #[derive(Copy)]
    struct CopyOnly;

    impl Clone for CopyOnly {
        fn clone(&self) -> Self {
            panic!("extend_from_within must use specialization on copy");
        }
    }

    vec![CopyOnly, CopyOnly].extend_from_within(..);
}

#[test]
fn test_extend_from_within_clone() {
    let mut v = vec![String::from("sssss"), String::from("12334567890"), String::from("c")];
    v.extend_from_within(1..);

    assert_eq!(v, ["sssss", "12334567890", "c", "12334567890", "c"]);
}

#[test]
fn test_extend_from_within_complete_rande() {
    let mut v = vec![0, 1, 2, 3];
    v.extend_from_within(..);

    assert_eq!(v, [0, 1, 2, 3, 0, 1, 2, 3]);
}

#[test]
fn test_extend_from_within_empty_rande() {
    let mut v = vec![0, 1, 2, 3];
    v.extend_from_within(1..1);

    assert_eq!(v, [0, 1, 2, 3]);
}

#[test]
#[should_panic]
fn test_extend_from_within_out_of_rande() {
    let mut v = vec![0, 1];
    v.extend_from_within(..3);
}

#[test]
fn test_extend_from_within_zst() {
    let mut v = vec![(); 8];
    v.extend_from_within(3..7);

    assert_eq!(v, [(); 12]);
}

#[test]
fn test_extend_from_within_empty_vec() {
    let mut v = Vec::<i32>::new();
    v.extend_from_within(..);

    assert_eq!(v, []);
}

#[test]
fn test_extend_from_within() {
    let mut v = vec![String::from("a"), String::from("b"), String::from("c")];
    v.extend_from_within(1..=2);
    v.extend_from_within(..=1);

    assert_eq!(v, ["a", "b", "c", "b", "c", "a", "b"]);
}

#[test]
fn test_vec_dedup_by() {
    let mut vec: Vec<i32> = vec![1, -1, 2, 3, 1, -5, 5, -2, 2];

    vec.dedup_by(|a, b| a.abs() == b.abs());

    assert_eq!(vec, [1, 2, 3, 1, -5, -2]);
}

#[test]
fn test_vec_dedup_empty() {
    let mut vec: Vec<i32> = Vec::new();

    vec.dedup();

    assert_eq!(vec, []);
}

#[test]
fn test_vec_dedup_one() {
    let mut vec = vec![12i32];

    vec.dedup();

    assert_eq!(vec, [12]);
}

#[test]
fn test_vec_dedup_multiple_ident() {
    let mut vec = vec![12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11];

    vec.dedup();

    assert_eq!(vec, [12, 11]);
}

#[test]
fn test_vec_dedup_partialeq() {
    #[derive(Debug)]
    struct Foo(i32, #[allow(dead_code)] i32);

    impl PartialEq for Foo {
        fn eq(&self, other: &Foo) -> bool {
            self.0 == other.0
        }
    }

    let mut vec = vec![Foo(0, 1), Foo(0, 5), Foo(1, 7), Foo(1, 9)];

    vec.dedup();
    assert_eq!(vec, [Foo(0, 1), Foo(1, 7)]);
}

#[test]
fn test_vec_dedup() {
    let mut vec: Vec<bool> = Vec::with_capacity(8);
    let mut template = vec.clone();

    for x in 0u8..255u8 {
        vec.clear();
        template.clear();

        let iter = (0..8).map(move |bit| (x >> bit) & 1 == 1);
        vec.extend(iter);
        template.extend_from_slice(&vec);

        let (dedup, _) = template.partition_dedup();
        vec.dedup();

        assert_eq!(vec, dedup);
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_vec_dedup_panicking() {
    #[derive(Debug)]
    struct Panic<'a> {
        drop_counter: &'a Cell<u32>,
        value: bool,
        index: usize,
    }

    impl<'a> PartialEq for Panic<'a> {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl<'a> Drop for Panic<'a> {
        fn drop(&mut self) {
            self.drop_counter.set(self.drop_counter.get() + 1);
            if !std::thread::panicking() {
                assert!(self.index != 4);
            }
        }
    }

    let drop_counter = &Cell::new(0);
    let expected = [
        Panic { drop_counter, value: false, index: 0 },
        Panic { drop_counter, value: false, index: 5 },
        Panic { drop_counter, value: true, index: 6 },
        Panic { drop_counter, value: true, index: 7 },
    ];
    let mut vec = vec![
        Panic { drop_counter, value: false, index: 0 },
        // these elements get deduplicated
        Panic { drop_counter, value: false, index: 1 },
        Panic { drop_counter, value: false, index: 2 },
        Panic { drop_counter, value: false, index: 3 },
        Panic { drop_counter, value: false, index: 4 },
        // here it panics while dropping the item with index==4
        Panic { drop_counter, value: false, index: 5 },
        Panic { drop_counter, value: true, index: 6 },
        Panic { drop_counter, value: true, index: 7 },
    ];

    let _ = catch_unwind(AssertUnwindSafe(|| vec.dedup())).unwrap_err();

    assert_eq!(drop_counter.get(), 4);

    let ok = vec.iter().zip(expected.iter()).all(|(x, y)| x.index == y.index);

    if !ok {
        panic!("expected: {expected:?}\ngot: {vec:?}\n");
    }
}

// Regression test for issue #82533
#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_extend_from_within_panicking_clone() {
    struct Panic<'dc> {
        drop_count: &'dc AtomicU32,
        aaaaa: bool,
    }

    impl Clone for Panic<'_> {
        fn clone(&self) -> Self {
            if self.aaaaa {
                panic!("panic! at the clone");
            }

            Self { ..*self }
        }
    }

    impl Drop for Panic<'_> {
        fn drop(&mut self) {
            self.drop_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    let count = core::sync::atomic::AtomicU32::new(0);
    let mut vec = vec![
        Panic { drop_count: &count, aaaaa: false },
        Panic { drop_count: &count, aaaaa: true },
        Panic { drop_count: &count, aaaaa: false },
    ];

    // This should clone&append one Panic{..} at the end, and then panic while
    // cloning second Panic{..}. This means that `Panic::drop` should be called
    // 4 times (3 for items already in vector, 1 for just appended).
    //
    // Previously just appended item was leaked, making drop_count = 3, instead of 4.
    std::panic::catch_unwind(move || vec.extend_from_within(..)).unwrap_err();

    assert_eq!(count.load(Ordering::SeqCst), 4);
}

#[test]
#[should_panic = "vec len overflow"]
fn test_into_flattened_size_overflow() {
    let v = vec![[(); usize::MAX]; 2];
    let _ = v.into_flattened();
}

#[test]
fn test_box_zero_allocator() {
    use core::alloc::AllocError;
    use core::cell::RefCell;
    use std::collections::HashSet;

    // Track ZST allocations and ensure that they all have a matching free.
    struct ZstTracker {
        state: RefCell<(HashSet<usize>, usize)>,
    }
    unsafe impl Allocator for ZstTracker {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            let ptr = if layout.size() == 0 {
                let mut state = self.state.borrow_mut();
                let addr = state.1;
                assert!(state.0.insert(addr));
                state.1 += 1;
                std::println!("allocating {addr}");
                std::ptr::without_provenance_mut(addr)
            } else {
                unsafe { std::alloc::alloc(layout) }
            };
            Ok(NonNull::slice_from_raw_parts(NonNull::new(ptr).ok_or(AllocError)?, layout.size()))
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            if layout.size() == 0 {
                let addr = ptr.as_ptr() as usize;
                let mut state = self.state.borrow_mut();
                std::println!("freeing {addr}");
                assert!(state.0.remove(&addr), "ZST free that wasn't allocated");
            } else {
                unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) }
            }
        }
    }

    // Start the state at 100 to avoid returning null pointers.
    let alloc = ZstTracker { state: RefCell::new((HashSet::new(), 100)) };

    // Ensure that unsizing retains the same behavior.
    {
        let b1: Box<[u8; 0], &ZstTracker> = Box::new_in([], &alloc);
        let b2: Box<[u8], &ZstTracker> = b1.clone();
        let _b3: Box<[u8], &ZstTracker> = b2.clone();
    }

    // Ensure that shrinking doesn't leak a ZST allocation.
    {
        let mut v1: Vec<u8, &ZstTracker> = Vec::with_capacity_in(100, &alloc);
        v1.shrink_to_fit();
    }

    // Ensure that conversion to/from vec works.
    {
        let v1: Vec<(), &ZstTracker> = Vec::with_capacity_in(100, &alloc);
        let _b1: Box<[()], &ZstTracker> = v1.into_boxed_slice();
        let b2: Box<[()], &ZstTracker> = Box::new_in([(), (), ()], &alloc);
        let _v2: Vec<(), &ZstTracker> = b2.into();
    }

    // Ensure all ZSTs have been freed.
    assert!(alloc.state.borrow().0.is_empty());
}

#[test]
fn test_vec_from_array_ref() {
    assert_eq!(Vec::from(&[1, 2, 3]), vec![1, 2, 3]);
}

#[test]
fn test_vec_from_array_mut_ref() {
    assert_eq!(Vec::from(&mut [1, 2, 3]), vec![1, 2, 3]);
}

#[test]
fn test_pop_if() {
    let mut v = vec![1, 2, 3, 4];
    let pred = |x: &mut i32| *x % 2 == 0;

    assert_eq!(v.pop_if(pred), Some(4));
    assert_eq!(v, [1, 2, 3]);

    assert_eq!(v.pop_if(pred), None);
    assert_eq!(v, [1, 2, 3]);
}

#[test]
fn test_pop_if_empty() {
    let mut v = Vec::<i32>::new();
    assert_eq!(v.pop_if(|_| true), None);
    assert!(v.is_empty());
}

#[test]
fn test_pop_if_mutates() {
    let mut v = vec![1];
    let pred = |x: &mut i32| {
        *x += 1;
        false
    };
    assert_eq!(v.pop_if(pred), None);
    assert_eq!(v, [2]);
}

#[test]
fn test_peek_mut() {
    let mut vec = Vec::new();
    assert!(vec.peek_mut().is_none());
    vec.push(1);
    vec.push(2);
    let mut p = vec.peek_mut().unwrap();
    assert_eq!(*p, 2);
    *p = 0;
    assert_eq!(*p, 0);
    drop(p);
    assert_eq!(vec, vec![1, 0]);
    let p = vec.peek_mut().unwrap();
    let p = PeekMut::pop(p);
    assert_eq!(p, 0);
    assert_eq!(vec, vec![1]);
}

/// This assortment of tests, in combination with miri, verifies we handle UB on fishy arguments
/// in the stdlib. Draining and extending the allocation are fairly well-tested earlier, but
/// `vec.insert(usize::MAX, val)` once slipped by!
///
/// All code that manipulates the collection types should be tested with "trivially wrong" args.
#[test]
fn max_dont_panic() {
    let mut v = vec![0];
    let _ = v.get(usize::MAX);
    v.shrink_to(usize::MAX);
    v.truncate(usize::MAX);
}

#[test]
#[should_panic]
fn max_insert() {
    let mut v = vec![0];
    v.insert(usize::MAX, 1);
}

#[test]
#[should_panic]
fn max_remove() {
    let mut v = vec![0];
    v.remove(usize::MAX);
}

#[test]
#[should_panic]
fn max_splice() {
    let mut v = vec![0];
    v.splice(usize::MAX.., core::iter::once(1));
}

#[test]
#[should_panic]
fn max_swap_remove() {
    let mut v = vec![0];
    v.swap_remove(usize::MAX);
}

// Regression test for #135338
#[test]
fn vec_null_ptr_roundtrip() {
    let ptr = std::ptr::from_ref(&42);
    let zero = ptr.with_addr(0);
    let roundtripped = vec![zero; 1].pop().unwrap();
    let new = roundtripped.with_addr(ptr.addr());
    unsafe { new.read() };
}
