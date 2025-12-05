//@ proc-macro: issue-104884.rs
//@ ignore-backends: gcc

use std::collections::BinaryHeap;

#[macro_use]
extern crate issue_104884;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct PriorityQueueEntry<T> {
    value: T,
}

#[derive(PartialOrd, AddImpl)]
//~^ ERROR can't compare `PriorityQueue<T>` with `PriorityQueue<T>`
//~| ERROR the trait bound `PriorityQueue<T>: Eq` is not satisfied
//~| ERROR can't compare `T` with `T`
//~| ERROR no method named `cmp` found for struct `BinaryHeap<PriorityQueueEntry<T>>`
//~| ERROR no field `height` on type `&PriorityQueue<T>`

struct PriorityQueue<T>(BinaryHeap<PriorityQueueEntry<T>>);
//~^ ERROR can't compare `BinaryHeap<PriorityQueueEntry<T>>` with `_`
fn main() {}
