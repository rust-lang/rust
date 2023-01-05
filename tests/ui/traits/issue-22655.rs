// run-pass
#![allow(dead_code)]
// Regression test for issue #22655: This test should not lead to
// infinite recursion.

// pretty-expanded FIXME #23616

unsafe impl<T: Send + ?Sized> Send for Unique<T> { }

pub struct Unique<T:?Sized> {
    pointer: *const T,
}

pub struct Node<V> {
    vals: V,
    edges: Unique<Node<V>>,
}

fn is_send<T: Send>() {}

fn main() {
    is_send::<Node<&'static ()>>();
}
