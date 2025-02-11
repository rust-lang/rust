//@ check-pass
// Used to overflow in `is_normalizable`

use std::marker::PhantomData;

struct Node<T: 'static> {
    m: PhantomData<&'static T>,
}

struct Digit<T> {
    elem: T,
}

enum FingerTree<T: 'static> {
    Single(T),

    Deep(Digit<T>, Box<FingerTree<Node<T>>>),
}

fn main() {}
