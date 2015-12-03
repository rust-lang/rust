// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::BinaryHeap;

#[test]
fn test_iterator() {
    let data = vec![5, 9, 3];
    let iterout = [9, 5, 3];
    let heap = BinaryHeap::from(data);
    let mut i = 0;
    for el in &heap {
        assert_eq!(*el, iterout[i]);
        i += 1;
    }
}

#[test]
fn test_iterator_reverse() {
    let data = vec![5, 9, 3];
    let iterout = vec![3, 5, 9];
    let pq = BinaryHeap::from(data);

    let v: Vec<_> = pq.iter().rev().cloned().collect();
    assert_eq!(v, iterout);
}

#[test]
fn test_move_iter() {
    let data = vec![5, 9, 3];
    let iterout = vec![9, 5, 3];
    let pq = BinaryHeap::from(data);

    let v: Vec<_> = pq.into_iter().collect();
    assert_eq!(v, iterout);
}

#[test]
fn test_move_iter_size_hint() {
    let data = vec![5, 9];
    let pq = BinaryHeap::from(data);

    let mut it = pq.into_iter();

    assert_eq!(it.size_hint(), (2, Some(2)));
    assert_eq!(it.next(), Some(9));

    assert_eq!(it.size_hint(), (1, Some(1)));
    assert_eq!(it.next(), Some(5));

    assert_eq!(it.size_hint(), (0, Some(0)));
    assert_eq!(it.next(), None);
}

#[test]
fn test_move_iter_reverse() {
    let data = vec![5, 9, 3];
    let iterout = vec![3, 5, 9];
    let pq = BinaryHeap::from(data);

    let v: Vec<_> = pq.into_iter().rev().collect();
    assert_eq!(v, iterout);
}

#[test]
fn test_peek_and_pop() {
    let data = vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
    let mut sorted = data.clone();
    sorted.sort();
    let mut heap = BinaryHeap::from(data);
    while !heap.is_empty() {
        assert_eq!(heap.peek().unwrap(), sorted.last().unwrap());
        assert_eq!(heap.pop().unwrap(), sorted.pop().unwrap());
    }
}

#[test]
fn test_push() {
    let mut heap = BinaryHeap::from(vec![2, 4, 9]);
    assert_eq!(heap.len(), 3);
    assert!(*heap.peek().unwrap() == 9);
    heap.push(11);
    assert_eq!(heap.len(), 4);
    assert!(*heap.peek().unwrap() == 11);
    heap.push(5);
    assert_eq!(heap.len(), 5);
    assert!(*heap.peek().unwrap() == 11);
    heap.push(27);
    assert_eq!(heap.len(), 6);
    assert!(*heap.peek().unwrap() == 27);
    heap.push(3);
    assert_eq!(heap.len(), 7);
    assert!(*heap.peek().unwrap() == 27);
    heap.push(103);
    assert_eq!(heap.len(), 8);
    assert!(*heap.peek().unwrap() == 103);
}

#[test]
fn test_push_unique() {
    let mut heap = BinaryHeap::<Box<_>>::from(vec![box 2, box 4, box 9]);
    assert_eq!(heap.len(), 3);
    assert!(*heap.peek().unwrap() == box 9);
    heap.push(box 11);
    assert_eq!(heap.len(), 4);
    assert!(*heap.peek().unwrap() == box 11);
    heap.push(box 5);
    assert_eq!(heap.len(), 5);
    assert!(*heap.peek().unwrap() == box 11);
    heap.push(box 27);
    assert_eq!(heap.len(), 6);
    assert!(*heap.peek().unwrap() == box 27);
    heap.push(box 3);
    assert_eq!(heap.len(), 7);
    assert!(*heap.peek().unwrap() == box 27);
    heap.push(box 103);
    assert_eq!(heap.len(), 8);
    assert!(*heap.peek().unwrap() == box 103);
}

#[test]
fn test_push_pop() {
    let mut heap = BinaryHeap::from(vec![5, 5, 2, 1, 3]);
    assert_eq!(heap.len(), 5);
    assert_eq!(heap.push_pop(6), 6);
    assert_eq!(heap.len(), 5);
    assert_eq!(heap.push_pop(0), 5);
    assert_eq!(heap.len(), 5);
    assert_eq!(heap.push_pop(4), 5);
    assert_eq!(heap.len(), 5);
    assert_eq!(heap.push_pop(1), 4);
    assert_eq!(heap.len(), 5);
}

#[test]
fn test_replace() {
    let mut heap = BinaryHeap::from(vec![5, 5, 2, 1, 3]);
    assert_eq!(heap.len(), 5);
    assert_eq!(heap.replace(6).unwrap(), 5);
    assert_eq!(heap.len(), 5);
    assert_eq!(heap.replace(0).unwrap(), 6);
    assert_eq!(heap.len(), 5);
    assert_eq!(heap.replace(4).unwrap(), 5);
    assert_eq!(heap.len(), 5);
    assert_eq!(heap.replace(1).unwrap(), 4);
    assert_eq!(heap.len(), 5);
}

fn check_to_vec(mut data: Vec<i32>) {
    let heap = BinaryHeap::from(data.clone());
    let mut v = heap.clone().into_vec();
    v.sort();
    data.sort();

    assert_eq!(v, data);
    assert_eq!(heap.into_sorted_vec(), data);
}

#[test]
fn test_to_vec() {
    check_to_vec(vec![]);
    check_to_vec(vec![5]);
    check_to_vec(vec![3, 2]);
    check_to_vec(vec![2, 3]);
    check_to_vec(vec![5, 1, 2]);
    check_to_vec(vec![1, 100, 2, 3]);
    check_to_vec(vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 0]);
    check_to_vec(vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1]);
    check_to_vec(vec![9, 11, 9, 9, 9, 9, 11, 2, 3, 4, 11, 9, 0, 0, 0, 0]);
    check_to_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    check_to_vec(vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    check_to_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 2]);
    check_to_vec(vec![5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1]);
}

#[test]
fn test_empty_pop() {
    let mut heap = BinaryHeap::<i32>::new();
    assert!(heap.pop().is_none());
}

#[test]
fn test_empty_peek() {
    let empty = BinaryHeap::<i32>::new();
    assert!(empty.peek().is_none());
}

#[test]
fn test_empty_replace() {
    let mut heap = BinaryHeap::new();
    assert!(heap.replace(5).is_none());
}

#[test]
fn test_from_iter() {
    let xs = vec![9, 8, 7, 6, 5, 4, 3, 2, 1];

    let mut q: BinaryHeap<_> = xs.iter().rev().cloned().collect();

    for &x in &xs {
        assert_eq!(q.pop().unwrap(), x);
    }
}

#[test]
fn test_drain() {
    let mut q: BinaryHeap<_> = [9, 8, 7, 6, 5, 4, 3, 2, 1].iter().cloned().collect();

    assert_eq!(q.drain().take(5).count(), 5);

    assert!(q.is_empty());
}

#[test]
fn test_extend_ref() {
    let mut a = BinaryHeap::new();
    a.push(1);
    a.push(2);

    a.extend(&[3, 4, 5]);

    assert_eq!(a.len(), 5);
    assert_eq!(a.into_sorted_vec(), [1, 2, 3, 4, 5]);

    let mut a = BinaryHeap::new();
    a.push(1);
    a.push(2);
    let mut b = BinaryHeap::new();
    b.push(3);
    b.push(4);
    b.push(5);

    a.extend(&b);

    assert_eq!(a.len(), 5);
    assert_eq!(a.into_sorted_vec(), [1, 2, 3, 4, 5]);
}
