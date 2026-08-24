//! Regression test for <https://github.com/rust-lang/rust/issues/48728>.
//! ICE occurred computing coherence "help" information.

//@ check-pass
#[derive(Clone)]
struct Node<T: ?Sized>(Box<T>);

impl<T: Clone + ?Sized> Clone for Node<[T]> {
    fn clone(&self) -> Self {
        Node(Box::clone(&self.0))
    }
}

fn main() {}
