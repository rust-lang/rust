#![allow(unused)]
#![warn(clippy::clear_with_drain)]

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

fn vec_range() {
    // Do not lint because iterator is assigned
    let mut v = vec![1, 2, 3];
    let iter = v.drain(0..v.len());

    // Do not lint because iterator is used
    let mut v = vec![1, 2, 3];
    let n = v.drain(0..v.len()).count();

    // Do not lint because iterator is assigned and used
    let mut v = vec![1, 2, 3];
    let iter = v.drain(usize::MIN..v.len());
    let n = iter.count();

    // Do lint
    let mut v = vec![1, 2, 3];
    v.drain(0..v.len());

    // Do lint
    let mut v = vec![1, 2, 3];
    v.drain(usize::MIN..v.len());
}

fn vec_range_from() {
    // Do not lint because iterator is assigned
    let mut v = vec![1, 2, 3];
    let iter = v.drain(0..);

    // Do not lint because iterator is assigned and used
    let mut v = vec![1, 2, 3];
    let mut iter = v.drain(0..);
    let next = iter.next();

    // Do not lint because iterator is used
    let mut v = vec![1, 2, 3];
    let next = v.drain(usize::MIN..).next();

    // Do lint
    let mut v = vec![1, 2, 3];
    v.drain(0..);

    // Do lint
    let mut v = vec![1, 2, 3];
    v.drain(usize::MIN..);
}

fn vec_range_full() {
    // Do not lint because iterator is assigned
    let mut v = vec![1, 2, 3];
    let iter = v.drain(..);

    // Do not lint because iterator is used
    let mut v = vec![1, 2, 3];
    for x in v.drain(..) {
        let y = format!("x = {x}");
    }

    // Do lint
    let mut v = vec![1, 2, 3];
    v.drain(..);
}

fn vec_range_to() {
    // Do not lint because iterator is assigned
    let mut v = vec![1, 2, 3];
    let iter = v.drain(..v.len());

    // Do not lint because iterator is assigned and used
    let mut v = vec![1, 2, 3];
    let iter = v.drain(..v.len());
    for x in iter {
        let y = format!("x = {x}");
    }

    // Do lint
    let mut v = vec![1, 2, 3];
    v.drain(..v.len());
}

fn vec_partial_drains() {
    // Do not lint any of these because the ranges are not full

    let mut v = vec![1, 2, 3];
    v.drain(1..);
    let mut v = vec![1, 2, 3];
    v.drain(1..).max();

    let mut v = vec![1, 2, 3];
    v.drain(..v.len() - 1);
    let mut v = vec![1, 2, 3];
    v.drain(..v.len() - 1).min();

    let mut v = vec![1, 2, 3];
    v.drain(1..v.len() - 1);
    let mut v = vec![1, 2, 3];
    let w: Vec<i8> = v.drain(1..v.len() - 1).collect();
}

fn vec_deque_range() {
    // Do not lint because iterator is assigned
    let mut deque = VecDeque::from([1, 2, 3]);
    let iter = deque.drain(0..deque.len());

    // Do not lint because iterator is used
    let mut deque = VecDeque::from([1, 2, 3]);
    let n = deque.drain(0..deque.len()).count();

    // Do not lint because iterator is assigned and used
    let mut deque = VecDeque::from([1, 2, 3]);
    let iter = deque.drain(usize::MIN..deque.len());
    let n = iter.count();

    // Do lint
    let mut deque = VecDeque::from([1, 2, 3]);
    deque.drain(0..deque.len());

    // Do lint
    let mut deque = VecDeque::from([1, 2, 3]);
    deque.drain(usize::MIN..deque.len());
}

fn vec_deque_range_from() {
    // Do not lint because iterator is assigned
    let mut deque = VecDeque::from([1, 2, 3]);
    let iter = deque.drain(0..);

    // Do not lint because iterator is assigned and used
    let mut deque = VecDeque::from([1, 2, 3]);
    let mut iter = deque.drain(0..);
    let next = iter.next();

    // Do not lint because iterator is used
    let mut deque = VecDeque::from([1, 2, 3]);
    let next = deque.drain(usize::MIN..).next();

    // Do lint
    let mut deque = VecDeque::from([1, 2, 3]);
    deque.drain(0..);

    // Do lint
    let mut deque = VecDeque::from([1, 2, 3]);
    deque.drain(usize::MIN..);
}

fn vec_deque_range_full() {
    // Do not lint because iterator is assigned
    let mut deque = VecDeque::from([1, 2, 3]);
    let iter = deque.drain(..);

    // Do not lint because iterator is used
    let mut deque = VecDeque::from([1, 2, 3]);
    for x in deque.drain(..) {
        let y = format!("x = {x}");
    }

    // Do lint
    let mut deque = VecDeque::from([1, 2, 3]);
    deque.drain(..);
}

fn vec_deque_range_to() {
    // Do not lint because iterator is assigned
    let mut deque = VecDeque::from([1, 2, 3]);
    let iter = deque.drain(..deque.len());

    // Do not lint because iterator is assigned and used
    let mut deque = VecDeque::from([1, 2, 3]);
    let iter = deque.drain(..deque.len());
    for x in iter {
        let y = format!("x = {x}");
    }

    // Do lint
    let mut deque = VecDeque::from([1, 2, 3]);
    deque.drain(..deque.len());
}

fn vec_deque_partial_drains() {
    // Do not lint any of these because the ranges are not full

    let mut deque = VecDeque::from([1, 2, 3]);
    deque.drain(1..);
    let mut deque = VecDeque::from([1, 2, 3]);
    deque.drain(1..).max();

    let mut deque = VecDeque::from([1, 2, 3]);
    deque.drain(..deque.len() - 1);
    let mut deque = VecDeque::from([1, 2, 3]);
    deque.drain(..deque.len() - 1).min();

    let mut deque = VecDeque::from([1, 2, 3]);
    deque.drain(1..deque.len() - 1);
    let mut deque = VecDeque::from([1, 2, 3]);
    let w: Vec<i8> = deque.drain(1..deque.len() - 1).collect();
}

fn string_range() {
    // Do not lint because iterator is assigned
    let mut s = String::from("Hello, world!");
    let iter = s.drain(0..s.len());

    // Do not lint because iterator is used
    let mut s = String::from("Hello, world!");
    let n = s.drain(0..s.len()).count();

    // Do not lint because iterator is assigned and used
    let mut s = String::from("Hello, world!");
    let iter = s.drain(usize::MIN..s.len());
    let n = iter.count();

    // Do lint
    let mut s = String::from("Hello, world!");
    s.drain(0..s.len());

    // Do lint
    let mut s = String::from("Hello, world!");
    s.drain(usize::MIN..s.len());
}

fn string_range_from() {
    // Do not lint because iterator is assigned
    let mut s = String::from("Hello, world!");
    let iter = s.drain(0..);

    // Do not lint because iterator is assigned and used
    let mut s = String::from("Hello, world!");
    let mut iter = s.drain(0..);
    let next = iter.next();

    // Do not lint because iterator is used
    let mut s = String::from("Hello, world!");
    let next = s.drain(usize::MIN..).next();

    // Do lint
    let mut s = String::from("Hello, world!");
    s.drain(0..);

    // Do lint
    let mut s = String::from("Hello, world!");
    s.drain(usize::MIN..);
}

fn string_range_full() {
    // Do not lint because iterator is assigned
    let mut s = String::from("Hello, world!");
    let iter = s.drain(..);

    // Do not lint because iterator is used
    let mut s = String::from("Hello, world!");
    for x in s.drain(..) {
        let y = format!("x = {x}");
    }

    // Do lint
    let mut s = String::from("Hello, world!");
    s.drain(..);
}

fn string_range_to() {
    // Do not lint because iterator is assigned
    let mut s = String::from("Hello, world!");
    let iter = s.drain(..s.len());

    // Do not lint because iterator is assigned and used
    let mut s = String::from("Hello, world!");
    let iter = s.drain(..s.len());
    for x in iter {
        let y = format!("x = {x}");
    }

    // Do lint
    let mut s = String::from("Hello, world!");
    s.drain(..s.len());
}

fn string_partial_drains() {
    // Do not lint any of these because the ranges are not full

    let mut s = String::from("Hello, world!");
    s.drain(1..);
    let mut s = String::from("Hello, world!");
    s.drain(1..).max();

    let mut s = String::from("Hello, world!");
    s.drain(..s.len() - 1);
    let mut s = String::from("Hello, world!");
    s.drain(..s.len() - 1).min();

    let mut s = String::from("Hello, world!");
    s.drain(1..s.len() - 1);
    let mut s = String::from("Hello, world!");
    let w: String = s.drain(1..s.len() - 1).collect();
}

fn hash_set() {
    // Do not lint because iterator is assigned
    let mut set = HashSet::from([1, 2, 3]);
    let iter = set.drain();

    // Do not lint because iterator is assigned and used
    let mut set = HashSet::from([1, 2, 3]);
    let mut iter = set.drain();
    let next = iter.next();

    // Do not lint because iterator is used
    let mut set = HashSet::from([1, 2, 3]);
    let next = set.drain().next();

    // Do lint
    let mut set = HashSet::from([1, 2, 3]);
    set.drain();
}

fn hash_map() {
    // Do not lint because iterator is assigned
    let mut map = HashMap::from([(1, "a"), (2, "b")]);
    let iter = map.drain();

    // Do not lint because iterator is assigned and used
    let mut map = HashMap::from([(1, "a"), (2, "b")]);
    let mut iter = map.drain();
    let next = iter.next();

    // Do not lint because iterator is used
    let mut map = HashMap::from([(1, "a"), (2, "b")]);
    let next = map.drain().next();

    // Do lint
    let mut map = HashMap::from([(1, "a"), (2, "b")]);
    map.drain();
}

fn binary_heap() {
    // Do not lint because iterator is assigned
    let mut heap = BinaryHeap::from([1, 2]);
    let iter = heap.drain();

    // Do not lint because iterator is assigned and used
    let mut heap = BinaryHeap::from([1, 2]);
    let mut iter = heap.drain();
    let next = iter.next();

    // Do not lint because iterator is used
    let mut heap = BinaryHeap::from([1, 2]);
    let next = heap.drain().next();

    // Do lint
    let mut heap = BinaryHeap::from([1, 2]);
    heap.drain();
}

fn main() {}
