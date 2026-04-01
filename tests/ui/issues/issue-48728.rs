// Regression test for #48728, an ICE that occurred computing
// coherence "help" information.

//@ check-pass
#[derive(Clone)]
struct Node<T: ?Sized>(Box<T>);

impl<T: Clone + ?Sized> Clone for Node<[T]> {
    fn clone(&self) -> Self {
        Node(Box::clone(&self.0))
    }
}

fn main() {}
