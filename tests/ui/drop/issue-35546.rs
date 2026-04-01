//@ build-pass
#![allow(dead_code)]
// Regression test for #35546. Check that we are able to codegen
// this. Before we had problems because of the drop glue signature
// around dropping a trait object (specifically, when dropping the
// `value` field of `Node<Send>`).

struct Node<T: ?Sized + Send> {
    next: Option<Box<Node<dyn Send>>>,
    value: T,
}

fn clear(head: &mut Option<Box<Node<dyn Send>>>) {
    match head.take() {
        Some(node) => *head = node.next,
        None => (),
    }
}

fn main() {}
