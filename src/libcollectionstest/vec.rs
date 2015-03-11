// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::iter::{FromIterator, repeat};
use std::mem::size_of;
use std::vec::as_vec;

use test::Bencher;

struct DropCounter<'a> {
    count: &'a mut u32
}

#[unsafe_destructor]
impl<'a> Drop for DropCounter<'a> {
    fn drop(&mut self) {
        *self.count += 1;
    }
}

#[test]
fn test_as_vec() {
    let xs = [1u8, 2u8, 3u8];
    assert_eq!(&**as_vec(&xs), xs);
}

#[test]
fn test_as_vec_dtor() {
    let (mut count_x, mut count_y) = (0, 0);
    {
        let xs = &[DropCounter { count: &mut count_x }, DropCounter { count: &mut count_y }];
        assert_eq!(as_vec(xs).len(), 2);
    }
    assert_eq!(count_x, 1);
    assert_eq!(count_y, 1);
}

#[test]
fn test_small_vec_struct() {
    assert!(size_of::<Vec<u8>>() == size_of::<usize>() * 3);
}

#[test]
fn test_double_drop() {
    struct TwoVec<T> {
        x: Vec<T>,
        y: Vec<T>
    }

    let (mut count_x, mut count_y) = (0, 0);
    {
        let mut tv = TwoVec {
            x: Vec::new(),
            y: Vec::new()
        };
        tv.x.push(DropCounter {count: &mut count_x});
        tv.y.push(DropCounter {count: &mut count_y});

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

    v.extend(0..3);
    for i in 0..3 { w.push(i) }

    assert_eq!(v, w);

    v.extend(3..10);
    for i in 3..10 { w.push(i) }

    assert_eq!(v, w);
}

#[test]
fn test_slice_from_mut() {
    let mut values = vec![1, 2, 3, 4, 5];
    {
        let slice = &mut values[2 ..];
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
        let slice = &mut values[.. 2];
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
    let w = vec!(1, 2, 3);

    assert_eq!(v, v.clone());

    let z = w.clone();
    assert_eq!(w, z);
    // they should be disjoint in memory.
    assert!(w.as_ptr() != z.as_ptr())
}

#[test]
fn test_clone_from() {
    let mut v = vec!();
    let three: Vec<Box<_>> = vec!(box 1, box 2, box 3);
    let two: Vec<Box<_>> = vec!(box 4, box 5);
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
    unsafe { v.set_len(0); }
    assert_eq!(v.iter_mut().count(), 0);
}

#[test]
fn test_partition() {
    assert_eq!(vec![].into_iter().partition(|x: &i32| *x < 3), (vec![], vec![]));
    assert_eq!(vec![1, 2, 3].into_iter().partition(|x| *x < 4), (vec![1, 2, 3], vec![]));
    assert_eq!(vec![1, 2, 3].into_iter().partition(|x| *x < 2), (vec![1], vec![2, 3]));
    assert_eq!(vec![1, 2, 3].into_iter().partition(|x| *x < 0), (vec![], vec![1, 2, 3]));
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
fn test_unsafe_ptrs() {
    unsafe {
        // Test on-stack copy-from-buf.
        let a = [1, 2, 3];
        let ptr = a.as_ptr();
        let b = Vec::from_raw_buf(ptr, 3);
        assert_eq!(b, [1, 2, 3]);

        // Test on-heap copy-from-buf.
        let c = vec![1, 2, 3, 4, 5];
        let ptr = c.as_ptr();
        let d = Vec::from_raw_buf(ptr, 5);
        assert_eq!(d, [1, 2, 3, 4, 5]);
    }
}

#[test]
fn test_vec_truncate_drop() {
    static mut drops: u32 = 0;
    struct Elem(i32);
    impl Drop for Elem {
        fn drop(&mut self) {
            unsafe { drops += 1; }
        }
    }

    let mut v = vec![Elem(1), Elem(2), Elem(3), Elem(4), Elem(5)];
    assert_eq!(unsafe { drops }, 0);
    v.truncate(3);
    assert_eq!(unsafe { drops }, 2);
    v.truncate(0);
    assert_eq!(unsafe { drops }, 5);
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
    &x[-1..];
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
    &x[-1..4];
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
    let mut vec= Vec::<i32>::new();
    vec.swap_remove(0);
}

#[test]
fn test_move_iter_unwrap() {
    let mut vec = Vec::with_capacity(7);
    vec.push(1);
    vec.push(2);
    let ptr = vec.as_ptr();
    vec = vec.into_iter().into_inner();
    assert_eq!(vec.as_ptr(), ptr);
    assert_eq!(vec.capacity(), 7);
    assert_eq!(vec.len(), 0);
}

#[test]
#[should_panic]
fn test_map_in_place_incompatible_types_fail() {
    let v = vec![0, 1, 2];
    v.map_in_place(|_| ());
}

#[test]
fn test_map_in_place() {
    let v = vec![0, 1, 2];
    assert_eq!(v.map_in_place(|i: u32| i as i32 - 1), [-1, 0, 1]);
}

#[test]
fn test_map_in_place_zero_sized() {
    let v = vec![(), ()];
    #[derive(PartialEq, Debug)]
    struct ZeroSized;
    assert_eq!(v.map_in_place(|_| ZeroSized), [ZeroSized, ZeroSized]);
}

#[test]
fn test_map_in_place_zero_drop_count() {
    use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};

    #[derive(Clone, PartialEq, Debug)]
    struct Nothing;
    impl Drop for Nothing { fn drop(&mut self) { } }

    #[derive(Clone, PartialEq, Debug)]
    struct ZeroSized;
    impl Drop for ZeroSized {
        fn drop(&mut self) {
            DROP_COUNTER.fetch_add(1, Ordering::Relaxed);
        }
    }
    const NUM_ELEMENTS: usize = 2;
    static DROP_COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;

    let v = repeat(Nothing).take(NUM_ELEMENTS).collect::<Vec<_>>();

    DROP_COUNTER.store(0, Ordering::Relaxed);

    let v = v.map_in_place(|_| ZeroSized);
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 0);
    drop(v);
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), NUM_ELEMENTS);
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
    for i in vec.drain() {
        vec2.push(i);
    }
    assert_eq!(vec, []);
    assert_eq!(vec2, [ 1, 2, 3 ]);
}

#[test]
fn test_drain_items_reverse() {
    let mut vec = vec![1, 2, 3];
    let mut vec2 = vec![];
    for i in vec.drain().rev() {
        vec2.push(i);
    }
    assert_eq!(vec, []);
    assert_eq!(vec2, [3, 2, 1]);
}

#[test]
fn test_drain_items_zero_sized() {
    let mut vec = vec![(), (), ()];
    let mut vec2 = vec![];
    for i in vec.drain() {
        vec2.push(i);
    }
    assert_eq!(vec, []);
    assert_eq!(vec2, [(), (), ()]);
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

#[bench]
fn bench_new(b: &mut Bencher) {
    b.iter(|| {
        let v: Vec<u32> = Vec::new();
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
    })
}

fn do_bench_with_capacity(b: &mut Bencher, src_len: usize) {
    b.bytes = src_len as u64;

    b.iter(|| {
        let v: Vec<u32> = Vec::with_capacity(src_len);
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), src_len);
    })
}

#[bench]
fn bench_with_capacity_0000(b: &mut Bencher) {
    do_bench_with_capacity(b, 0)
}

#[bench]
fn bench_with_capacity_0010(b: &mut Bencher) {
    do_bench_with_capacity(b, 10)
}

#[bench]
fn bench_with_capacity_0100(b: &mut Bencher) {
    do_bench_with_capacity(b, 100)
}

#[bench]
fn bench_with_capacity_1000(b: &mut Bencher) {
    do_bench_with_capacity(b, 1000)
}

fn do_bench_from_fn(b: &mut Bencher, src_len: usize) {
    b.bytes = src_len as u64;

    b.iter(|| {
        let dst = (0..src_len).collect::<Vec<_>>();
        assert_eq!(dst.len(), src_len);
        assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
    })
}

#[bench]
fn bench_from_fn_0000(b: &mut Bencher) {
    do_bench_from_fn(b, 0)
}

#[bench]
fn bench_from_fn_0010(b: &mut Bencher) {
    do_bench_from_fn(b, 10)
}

#[bench]
fn bench_from_fn_0100(b: &mut Bencher) {
    do_bench_from_fn(b, 100)
}

#[bench]
fn bench_from_fn_1000(b: &mut Bencher) {
    do_bench_from_fn(b, 1000)
}

fn do_bench_from_elem(b: &mut Bencher, src_len: usize) {
    b.bytes = src_len as u64;

    b.iter(|| {
        let dst: Vec<usize> = repeat(5).take(src_len).collect();
        assert_eq!(dst.len(), src_len);
        assert!(dst.iter().all(|x| *x == 5));
    })
}

#[bench]
fn bench_from_elem_0000(b: &mut Bencher) {
    do_bench_from_elem(b, 0)
}

#[bench]
fn bench_from_elem_0010(b: &mut Bencher) {
    do_bench_from_elem(b, 10)
}

#[bench]
fn bench_from_elem_0100(b: &mut Bencher) {
    do_bench_from_elem(b, 100)
}

#[bench]
fn bench_from_elem_1000(b: &mut Bencher) {
    do_bench_from_elem(b, 1000)
}

fn do_bench_from_slice(b: &mut Bencher, src_len: usize) {
    let src: Vec<_> = FromIterator::from_iter(0..src_len);

    b.bytes = src_len as u64;

    b.iter(|| {
        let dst = src.clone()[..].to_vec();
        assert_eq!(dst.len(), src_len);
        assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
    });
}

#[bench]
fn bench_from_slice_0000(b: &mut Bencher) {
    do_bench_from_slice(b, 0)
}

#[bench]
fn bench_from_slice_0010(b: &mut Bencher) {
    do_bench_from_slice(b, 10)
}

#[bench]
fn bench_from_slice_0100(b: &mut Bencher) {
    do_bench_from_slice(b, 100)
}

#[bench]
fn bench_from_slice_1000(b: &mut Bencher) {
    do_bench_from_slice(b, 1000)
}

fn do_bench_from_iter(b: &mut Bencher, src_len: usize) {
    let src: Vec<_> = FromIterator::from_iter(0..src_len);

    b.bytes = src_len as u64;

    b.iter(|| {
        let dst: Vec<_> = FromIterator::from_iter(src.clone().into_iter());
        assert_eq!(dst.len(), src_len);
        assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
    });
}

#[bench]
fn bench_from_iter_0000(b: &mut Bencher) {
    do_bench_from_iter(b, 0)
}

#[bench]
fn bench_from_iter_0010(b: &mut Bencher) {
    do_bench_from_iter(b, 10)
}

#[bench]
fn bench_from_iter_0100(b: &mut Bencher) {
    do_bench_from_iter(b, 100)
}

#[bench]
fn bench_from_iter_1000(b: &mut Bencher) {
    do_bench_from_iter(b, 1000)
}

fn do_bench_extend(b: &mut Bencher, dst_len: usize, src_len: usize) {
    let dst: Vec<_> = FromIterator::from_iter(0..dst_len);
    let src: Vec<_> = FromIterator::from_iter(dst_len..dst_len + src_len);

    b.bytes = src_len as u64;

    b.iter(|| {
        let mut dst = dst.clone();
        dst.extend(src.clone().into_iter());
        assert_eq!(dst.len(), dst_len + src_len);
        assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
    });
}

#[bench]
fn bench_extend_0000_0000(b: &mut Bencher) {
    do_bench_extend(b, 0, 0)
}

#[bench]
fn bench_extend_0000_0010(b: &mut Bencher) {
    do_bench_extend(b, 0, 10)
}

#[bench]
fn bench_extend_0000_0100(b: &mut Bencher) {
    do_bench_extend(b, 0, 100)
}

#[bench]
fn bench_extend_0000_1000(b: &mut Bencher) {
    do_bench_extend(b, 0, 1000)
}

#[bench]
fn bench_extend_0010_0010(b: &mut Bencher) {
    do_bench_extend(b, 10, 10)
}

#[bench]
fn bench_extend_0100_0100(b: &mut Bencher) {
    do_bench_extend(b, 100, 100)
}

#[bench]
fn bench_extend_1000_1000(b: &mut Bencher) {
    do_bench_extend(b, 1000, 1000)
}

fn do_bench_push_all(b: &mut Bencher, dst_len: usize, src_len: usize) {
    let dst: Vec<_> = FromIterator::from_iter(0..dst_len);
    let src: Vec<_> = FromIterator::from_iter(dst_len..dst_len + src_len);

    b.bytes = src_len as u64;

    b.iter(|| {
        let mut dst = dst.clone();
        dst.push_all(&src);
        assert_eq!(dst.len(), dst_len + src_len);
        assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
    });
}

#[bench]
fn bench_push_all_0000_0000(b: &mut Bencher) {
    do_bench_push_all(b, 0, 0)
}

#[bench]
fn bench_push_all_0000_0010(b: &mut Bencher) {
    do_bench_push_all(b, 0, 10)
}

#[bench]
fn bench_push_all_0000_0100(b: &mut Bencher) {
    do_bench_push_all(b, 0, 100)
}

#[bench]
fn bench_push_all_0000_1000(b: &mut Bencher) {
    do_bench_push_all(b, 0, 1000)
}

#[bench]
fn bench_push_all_0010_0010(b: &mut Bencher) {
    do_bench_push_all(b, 10, 10)
}

#[bench]
fn bench_push_all_0100_0100(b: &mut Bencher) {
    do_bench_push_all(b, 100, 100)
}

#[bench]
fn bench_push_all_1000_1000(b: &mut Bencher) {
    do_bench_push_all(b, 1000, 1000)
}

fn do_bench_push_all_move(b: &mut Bencher, dst_len: usize, src_len: usize) {
    let dst: Vec<_> = FromIterator::from_iter(0..dst_len);
    let src: Vec<_> = FromIterator::from_iter(dst_len..dst_len + src_len);

    b.bytes = src_len as u64;

    b.iter(|| {
        let mut dst = dst.clone();
        dst.extend(src.clone().into_iter());
        assert_eq!(dst.len(), dst_len + src_len);
        assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
    });
}

#[bench]
fn bench_push_all_move_0000_0000(b: &mut Bencher) {
    do_bench_push_all_move(b, 0, 0)
}

#[bench]
fn bench_push_all_move_0000_0010(b: &mut Bencher) {
    do_bench_push_all_move(b, 0, 10)
}

#[bench]
fn bench_push_all_move_0000_0100(b: &mut Bencher) {
    do_bench_push_all_move(b, 0, 100)
}

#[bench]
fn bench_push_all_move_0000_1000(b: &mut Bencher) {
    do_bench_push_all_move(b, 0, 1000)
}

#[bench]
fn bench_push_all_move_0010_0010(b: &mut Bencher) {
    do_bench_push_all_move(b, 10, 10)
}

#[bench]
fn bench_push_all_move_0100_0100(b: &mut Bencher) {
    do_bench_push_all_move(b, 100, 100)
}

#[bench]
fn bench_push_all_move_1000_1000(b: &mut Bencher) {
    do_bench_push_all_move(b, 1000, 1000)
}

fn do_bench_clone(b: &mut Bencher, src_len: usize) {
    let src: Vec<usize> = FromIterator::from_iter(0..src_len);

    b.bytes = src_len as u64;

    b.iter(|| {
        let dst = src.clone();
        assert_eq!(dst.len(), src_len);
        assert!(dst.iter().enumerate().all(|(i, x)| i == *x));
    });
}

#[bench]
fn bench_clone_0000(b: &mut Bencher) {
    do_bench_clone(b, 0)
}

#[bench]
fn bench_clone_0010(b: &mut Bencher) {
    do_bench_clone(b, 10)
}

#[bench]
fn bench_clone_0100(b: &mut Bencher) {
    do_bench_clone(b, 100)
}

#[bench]
fn bench_clone_1000(b: &mut Bencher) {
    do_bench_clone(b, 1000)
}

fn do_bench_clone_from(b: &mut Bencher, times: usize, dst_len: usize, src_len: usize) {
    let dst: Vec<_> = FromIterator::from_iter(0..src_len);
    let src: Vec<_> = FromIterator::from_iter(dst_len..dst_len + src_len);

    b.bytes = (times * src_len) as u64;

    b.iter(|| {
        let mut dst = dst.clone();

        for _ in 0..times {
            dst.clone_from(&src);

            assert_eq!(dst.len(), src_len);
            assert!(dst.iter().enumerate().all(|(i, x)| dst_len + i == *x));
        }
    });
}

#[bench]
fn bench_clone_from_01_0000_0000(b: &mut Bencher) {
    do_bench_clone_from(b, 1, 0, 0)
}

#[bench]
fn bench_clone_from_01_0000_0010(b: &mut Bencher) {
    do_bench_clone_from(b, 1, 0, 10)
}

#[bench]
fn bench_clone_from_01_0000_0100(b: &mut Bencher) {
    do_bench_clone_from(b, 1, 0, 100)
}

#[bench]
fn bench_clone_from_01_0000_1000(b: &mut Bencher) {
    do_bench_clone_from(b, 1, 0, 1000)
}

#[bench]
fn bench_clone_from_01_0010_0010(b: &mut Bencher) {
    do_bench_clone_from(b, 1, 10, 10)
}

#[bench]
fn bench_clone_from_01_0100_0100(b: &mut Bencher) {
    do_bench_clone_from(b, 1, 100, 100)
}

#[bench]
fn bench_clone_from_01_1000_1000(b: &mut Bencher) {
    do_bench_clone_from(b, 1, 1000, 1000)
}

#[bench]
fn bench_clone_from_01_0010_0100(b: &mut Bencher) {
    do_bench_clone_from(b, 1, 10, 100)
}

#[bench]
fn bench_clone_from_01_0100_1000(b: &mut Bencher) {
    do_bench_clone_from(b, 1, 100, 1000)
}

#[bench]
fn bench_clone_from_01_0010_0000(b: &mut Bencher) {
    do_bench_clone_from(b, 1, 10, 0)
}

#[bench]
fn bench_clone_from_01_0100_0010(b: &mut Bencher) {
    do_bench_clone_from(b, 1, 100, 10)
}

#[bench]
fn bench_clone_from_01_1000_0100(b: &mut Bencher) {
    do_bench_clone_from(b, 1, 1000, 100)
}

#[bench]
fn bench_clone_from_10_0000_0000(b: &mut Bencher) {
    do_bench_clone_from(b, 10, 0, 0)
}

#[bench]
fn bench_clone_from_10_0000_0010(b: &mut Bencher) {
    do_bench_clone_from(b, 10, 0, 10)
}

#[bench]
fn bench_clone_from_10_0000_0100(b: &mut Bencher) {
    do_bench_clone_from(b, 10, 0, 100)
}

#[bench]
fn bench_clone_from_10_0000_1000(b: &mut Bencher) {
    do_bench_clone_from(b, 10, 0, 1000)
}

#[bench]
fn bench_clone_from_10_0010_0010(b: &mut Bencher) {
    do_bench_clone_from(b, 10, 10, 10)
}

#[bench]
fn bench_clone_from_10_0100_0100(b: &mut Bencher) {
    do_bench_clone_from(b, 10, 100, 100)
}

#[bench]
fn bench_clone_from_10_1000_1000(b: &mut Bencher) {
    do_bench_clone_from(b, 10, 1000, 1000)
}

#[bench]
fn bench_clone_from_10_0010_0100(b: &mut Bencher) {
    do_bench_clone_from(b, 10, 10, 100)
}

#[bench]
fn bench_clone_from_10_0100_1000(b: &mut Bencher) {
    do_bench_clone_from(b, 10, 100, 1000)
}

#[bench]
fn bench_clone_from_10_0010_0000(b: &mut Bencher) {
    do_bench_clone_from(b, 10, 10, 0)
}

#[bench]
fn bench_clone_from_10_0100_0010(b: &mut Bencher) {
    do_bench_clone_from(b, 10, 100, 10)
}

#[bench]
fn bench_clone_from_10_1000_0100(b: &mut Bencher) {
    do_bench_clone_from(b, 10, 1000, 100)
}
