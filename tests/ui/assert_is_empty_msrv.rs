#![warn(clippy::assert_is_empty)]

use std::collections::BinaryHeap;

#[clippy::msrv = "1.79"]
fn below_msrv(heap: BinaryHeap<u8>, borrowed: &&mut BinaryHeap<u8>, items: Vec<u8>) {
    assert!(heap.is_empty());
    assert!(!heap.is_empty());
    debug_assert!(borrowed.is_empty());
    debug_assert!(!borrowed.is_empty());

    // Older collection suggestions are still available below Rust 1.80.
    assert!(items.is_empty());
    //~^ assert_is_empty
}

#[clippy::msrv = "1.80"]
fn at_msrv(heap: BinaryHeap<u8>, borrowed: &&mut BinaryHeap<u8>) {
    assert!(heap.is_empty());
    //~^ assert_is_empty
    assert!(!heap.is_empty());
    //~^ assert_is_empty
    debug_assert!(borrowed.is_empty());
    //~^ assert_is_empty
    debug_assert!(!borrowed.is_empty());
    //~^ assert_is_empty
}

fn main() {}
