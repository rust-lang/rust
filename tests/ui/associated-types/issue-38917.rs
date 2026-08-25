//@ check-pass

use std::borrow::Borrow;

trait TNode: Sized {
    type ConcreteElement: TElement<ConcreteNode = Self>;
}

trait TElement: Sized {
    type ConcreteNode: TNode<ConcreteElement = Self>;
}

trait DomTraversal<N: TNode> {
    type BorrowElement: Borrow<N::ConcreteElement>;
}

#[allow(dead_code)]
fn recalc_style_at<E, D>()
where
    E: TElement,
    D: DomTraversal<E::ConcreteNode>,
{
}

fn main() {}
