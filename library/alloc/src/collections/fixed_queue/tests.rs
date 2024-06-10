use super::*;
use crate::{string::String, vec::Vec};

#[test]
fn with() {
    let x = FixedQueue::<usize, 3>::with([None; 3], 789, 456, 123);
    let y = FixedQueue::<usize, 3> { buffer: [None; 3], head: 789, tail: 456, len: 123 };
    assert_eq!(x, y);
}

#[test]
fn partial_eq_self() {
    let x = FixedQueue::<usize, 3>::new();
    let mut y = FixedQueue::with([None; 3], 0, 0, 0);
    assert_eq!(x, y);
    y.push(1);
    assert_ne!(x, y);
    y.push(2);
    assert_ne!(x, y);
    y.push(3);
    assert_ne!(x, y);
    y.clear();
    assert_eq!(x, y);
    let w = FixedQueue::<usize, 3>::with([None; 3], 0, 0, 0);
    let z = FixedQueue::<usize, 3>::with([None; 3], 1, 1, 0);
    assert_eq!(w, z);
    let u = FixedQueue::<usize, 3>::with([Some(20), None, None], 0, 1, 1);
    let v = FixedQueue::<usize, 3>::with([None, Some(20), None], 1, 2, 1);
    assert_eq!(u, v);
}

#[test]
fn partial_eq_array() {
    let x = FixedQueue::<usize, 3>::from([1, 2, 3]);
    assert_eq!(x, [1, 2, 3]);
    assert_ne!(x, [20, 2, 3]);
    let y = FixedQueue::<usize, 1>::from([80]);
    assert_eq!(y, [80]);
    let z = FixedQueue::<usize, 3>::with([Some(1), Some(2), Some(3)], 1, 1, 3);
    assert_eq!(z, [2, 3, 1]);
    let w = FixedQueue::<usize, 3>::with([Some(20), None, None], 0, 1, 1);
    assert_eq!(w, [20]);
    let u = FixedQueue::<usize, 3>::with([None, Some(20), None], 1, 2, 1);
    assert_eq!(u, [20]);
}

#[test]
fn new() {
    let x = FixedQueue::<usize, 3>::new();
    let y = FixedQueue::<usize, 3>::with([None; 3], 0, 0, 0);
    assert_eq!(x, y);
}

#[test]
fn from_array() {
    let x = FixedQueue::from([1i32, 2i32, 3i32]);
    let y = FixedQueue::<i32, 3>::with([Some(1i32), Some(2i32), Some(3i32)], 0, 0, 3);
    assert_eq!(x, y);
    let z = FixedQueue::from([true, false, true]);
    let w = FixedQueue::<bool, 3>::with([Some(true), Some(false), Some(true)], 0, 0, 3);
    assert_eq!(z, w);
}

#[test]
fn from_sized_slice() {
    let x = FixedQueue::from(&[3i32, 2i32, 1i32]);
    let y = FixedQueue::<i32, 3>::with([Some(3i32), Some(2i32), Some(1i32)], 0, 0, 3);
    assert_eq!(x, y);
}

#[test]
fn from_slice() {
    let array = [3i32, 2i32, 1i32];
    let x = FixedQueue::<i32, 1>::from(&array[0..1]);
    let y = FixedQueue::<i32, 1>::with([Some(3i32)], 0, 0, 1);
    assert_eq!(x, y);
    let w = FixedQueue::<i32, 2>::from(&array[0..2]);
    let z = FixedQueue::<i32, 2>::with([Some(3i32), Some(2i32)], 0, 0, 2);
    assert_eq!(w, z);
    let u = FixedQueue::<i32, 3>::from(&array[0..3]);
    let v = FixedQueue::<i32, 3>::with([Some(3i32), Some(2i32), Some(1i32)], 0, 0, 3);
    assert_eq!(u, v);
    let s = FixedQueue::<i32, 3>::from(&array[..]);
    let t = FixedQueue::<i32, 3>::with([Some(3i32), Some(2i32), Some(1i32)], 0, 0, 3);
    assert_eq!(s, t);
}

#[test]
fn index() {
    let x = FixedQueue::<&str, 3>::from(["a", "b", "c"]);
    assert_eq!(x[0], Some("a"));
    assert_eq!(x[1], Some("b"));
    assert_eq!(x[2], Some("c"));
}

#[test]
fn index_range() {
    let x = FixedQueue::<&str, 3>::from(["a", "b", "c"]);
    assert!(x[0..0].is_empty());
    assert_eq!(x[0..1], ["a"]);
    assert_eq!(x[0..2], ["a", "b"]);
    assert_eq!(x[0..3], ["a", "b", "c"]);
}

#[test]
fn display() {
    let mut x = FixedQueue::<usize, 3>::new();
    assert_eq!(format!("{}", x), String::from("{}"));
    x.push(10);
    assert_eq!(format!("{}", x), String::from("{10}"));
    x.pop();
    assert_eq!(format!("{}", x), String::from("{}"));
    x.push(20);
    assert_eq!(format!("{}", x), String::from("{20}"));
    x.push(30);
    assert_eq!(format!("{}", x), String::from("{20, 30}"));
    x.push(40);
    assert_eq!(format!("{}", x), String::from("{20, 30, 40}"));
    x.push(50);
    assert_eq!(format!("{}", x), String::from("{30, 40, 50}"));
    x.pop();
    assert_eq!(format!("{}", x), String::from("{40, 50}"));
    x.pop();
    assert_eq!(format!("{}", x), String::from("{50}"));
    x.pop();
    assert_eq!(format!("{}", x), String::from("{}"));
}

#[test]
fn capacity() {
    let x = FixedQueue::<usize, 1>::new();
    assert_eq!(x.capacity(), 1);
    let y = FixedQueue::<usize, 2>::new();
    assert_eq!(y.capacity(), 2);
    let z = FixedQueue::<usize, 3>::new();
    assert_eq!(z.capacity(), 3);
}

#[test]
fn len() {
    let mut x = FixedQueue::<bool, 3>::new();
    assert_eq!(x.len(), 0);
    x.push(true);
    assert_eq!(x.len(), 1);
    x.push(false);
    assert_eq!(x.len(), 2);
    x.push(true);
    assert_eq!(x.len(), 3);
    x.pop();
    assert_eq!(x.len(), 2);
    x.pop();
    assert_eq!(x.len(), 1);
    x.pop();
    assert_eq!(x.len(), 0);
}

#[test]
fn is_empty() {
    let mut x = FixedQueue::<usize, 3>::new();
    assert!(x.is_empty());
    x.push(1);
    assert!(!x.is_empty());
}

#[test]
fn is_full() {
    let mut x = FixedQueue::<usize, 3>::new();
    assert!(!x.is_full());
    x.push(1);
    assert!(!x.is_full());
    x.push(1);
    assert!(!x.is_full());
    x.push(1);
    assert!(x.is_full());
}

#[test]
fn clear() {
    let mut x = FixedQueue::from([1, 2, 3]);
    assert!(!x.is_empty());
    x.clear();
    assert!(x.is_empty());
}

#[test]
fn fill() {
    let mut x = FixedQueue::<usize, 3>::new();
    assert!(!x.is_full());
    x.fill(10);
    assert!(x.is_full());
}

#[test]
fn push() {
    let mut x = FixedQueue::<String, 2>::from([(); 2].map(|_| String::new()));
    assert!(x.is_full());
    x.clear();
    assert!(x.is_empty());
    x.push(String::from("a"));
    assert_eq!(x.len(), 1);
    assert_eq!(x, [String::from("a")]);
    x.push(String::from("b"));
    assert_eq!(x.len(), 2);
    assert_eq!(x, [String::from("a"), String::from("b")]);
    x.push(String::from("c"));
    assert_eq!(x.len(), 2);
    assert_eq!(x, [String::from("b"), String::from("c")]);
    x.push(String::from("d"));
    assert_eq!(x.len(), 2);
    assert_eq!(x, [String::from("c"), String::from("d")]);
    x.push(String::from("e"));
    assert_eq!(x.len(), 2);
    assert_eq!(x, [String::from("d"), String::from("e")]);
}

#[test]
fn pop() {
    let mut u = FixedQueue::<&str, 3>::from(["a", "b", "c"]);
    assert!(!u.is_empty());
    let w = u.pop();
    assert_eq!(w, Some("a"));
    assert_eq!(u, ["b", "c"]);
    let x = u.pop();
    assert_eq!(x, Some("b"));
    assert_eq!(u, ["c"]);
    let y = u.pop();
    assert_eq!(y, Some("c"));
    assert_eq!(u, []);
    let z = u.pop();
    assert_eq!(z, None);
    assert_eq!(u, []);
}

#[test]
fn to_option_array() {
    let x = FixedQueue::<&str, 3>::from(["a", "b", "c"]);
    assert_eq!(x.to_option_array(), [Some("a"), Some("b"), Some("c")]);
    let mut y = FixedQueue::<&str, 3>::from(["a", "b", "c"]);
    y.pop();
    assert_eq!(y.to_option_array(), [Some("b"), Some("c"), None])
}

#[test]
fn to_vec() {
    let x = FixedQueue::<&str, 3>::from(["a", "b", "c"]);
    let y = Vec::from(["a", "b", "c"]);
    assert_eq!(x.to_vec(), y);
}
