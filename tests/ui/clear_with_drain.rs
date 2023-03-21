#![allow(unused)]
#![warn(clippy::clear_with_drain)]

fn range() {
    let (mut u, mut v) = (vec![1, 2, 3], vec![1, 2, 3]);
    let iter = u.drain(0..u.len()); // Yay
    v.drain(0..v.len()); // Nay
}

fn range_from() {
    let (mut u, mut v) = (vec![1, 2, 3], vec![1, 2, 3]);
    let iter = u.drain(0..); // Yay
    v.drain(0..); // Nay
}

fn range_full() {
    let (mut u, mut v) = (vec![1, 2, 3], vec![1, 2, 3]);
    let iter = u.drain(..); // Yay
    v.drain(..); // Nay
}

fn range_to() {
    let (mut u, mut v) = (vec![1, 2, 3], vec![1, 2, 3]);
    let iter = u.drain(..u.len()); // Yay
    v.drain(..v.len()); // Nay
}

fn partial_drains() {
    let mut v = vec![1, 2, 3];
    v.drain(1..); // Yay

    let mut v = vec![1, 2, 3];
    v.drain(..v.len() - 1); // Yay

    let mut v = vec![1, 2, 3];
    v.drain(1..v.len() - 1); // Yay
}

fn main() {}
