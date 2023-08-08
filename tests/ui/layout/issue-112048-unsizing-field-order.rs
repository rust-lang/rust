// run-pass

// Check that unsizing doesn't reorder fields.

#![allow(dead_code)]

use std::fmt::Debug;

#[derive(Debug)]
struct GcNode<T: ?Sized> {
    gets_swapped_with_next: usize,
    next: Option<&'static GcNode<dyn Debug>>,
    tail: T,
}

fn main() {
    let node: Box<GcNode<dyn Debug>> = Box::new(GcNode {
        gets_swapped_with_next: 42,
        next: None,
        tail: Box::new(1),
    });

    assert_eq!(node.gets_swapped_with_next, 42);
    assert!(node.next.is_none());
}
