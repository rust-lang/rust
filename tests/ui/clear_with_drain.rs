// run-rustfix
#![allow(unused)]
#![warn(clippy::clear_with_drain)]

use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

fn range() {
    let mut v = vec![1, 2, 3];
    let iter = v.drain(0..v.len()); // Yay

    let mut v = vec![1, 2, 3];
    let n = v.drain(0..v.len()).count(); // Yay

    let mut v = vec![1, 2, 3];
    let iter = v.drain(usize::MIN..v.len()); // Yay
    let n = iter.count();

    let mut v = vec![1, 2, 3];
    v.drain(0..v.len()); // Nay

    let mut v = vec![1, 2, 3];
    v.drain(usize::MIN..v.len()); // Nay
}

fn range_from() {
    let mut v = vec![1, 2, 3];
    let iter = v.drain(0..); // Yay

    let mut v = vec![1, 2, 3];
    let mut iter = v.drain(0..); // Yay
    let next = iter.next();

    let mut v = vec![1, 2, 3];
    let next = v.drain(usize::MIN..).next(); // Yay

    let mut v = vec![1, 2, 3];
    v.drain(0..); // Nay

    let mut v = vec![1, 2, 3];
    v.drain(usize::MIN..); // Nay
}

fn range_full() {
    let mut v = vec![1, 2, 3];
    let iter = v.drain(..); // Yay

    let mut v = vec![1, 2, 3];
    // Yay
    for x in v.drain(..) {
        let y = format!("x = {x}");
    }

    let mut v = vec![1, 2, 3];
    v.drain(..); // Nay
}

fn range_to() {
    let mut v = vec![1, 2, 3];
    let iter = v.drain(..v.len()); // Yay

    let mut v = vec![1, 2, 3];
    let iter = v.drain(..v.len()); // Yay
    for x in iter {
        let y = format!("x = {x}");
    }

    let mut v = vec![1, 2, 3];
    v.drain(..v.len()); // Nay
}

fn partial_drains() {
    let mut v = vec![1, 2, 3];
    v.drain(1..); // Yay
    let mut v = vec![1, 2, 3];
    v.drain(1..).max(); // Yay

    let mut v = vec![1, 2, 3];
    v.drain(..v.len() - 1); // Yay
    let mut v = vec![1, 2, 3];
    v.drain(..v.len() - 1).min(); // Yay

    let mut v = vec![1, 2, 3];
    v.drain(1..v.len() - 1); // Yay
    let mut v = vec![1, 2, 3];
    let w: Vec<i8> = v.drain(1..v.len() - 1).collect(); // Yay
}

fn main() {
    let mut deque: VecDeque<_> = [1, 2, 3].into();
    deque.drain(..);

    let mut set = HashSet::from([1, 2, 3]);
    set.drain();

    let mut a = HashMap::new();
    a.insert(1, "a");
    a.insert(2, "b");
    a.drain();

    let mut heap = BinaryHeap::from([1, 3]);
    heap.drain();

    // Not firing for now because `String` is not reckognized by `is_type_diagnostic_item`
    let mut s = String::from("α is alpha, β is beta");
    s.drain(..);
}
