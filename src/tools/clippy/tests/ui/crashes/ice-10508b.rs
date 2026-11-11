//@ check-pass

use std::marker::PhantomData;

struct Digit<T> {
    elem: T,
}

struct Node<T: 'static> {
    m: PhantomData<&'static T>,
}

enum FingerTree<T: 'static> {
    Single(T),

    Deep(Digit<T>, Node<FingerTree<Node<T>>>),
}

enum Wrapper<T: 'static> {
    Simple,
    Other(FingerTree<T>),
}

fn main() {
    let w = Some(Wrapper::Simple::<u32>);
}
