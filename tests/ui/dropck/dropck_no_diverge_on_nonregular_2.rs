// Issue 22443: Reject code using non-regular types that would
// otherwise cause dropck to loop infinitely.

use std::marker::PhantomData;

struct Digit<T> {
    elem: T
}

struct Node<T:'static> { m: PhantomData<&'static T> }

enum FingerTree<T:'static> {
    Single(T),
    // Bug report said Digit before Box would infinite loop (versus
    // Digit after Box; see dropck_no_diverge_on_nonregular_1).
    Deep(
        Digit<T>,
        Box<FingerTree<Node<T>>>,
        )
}

fn main() {
    let ft = //~ ERROR overflow while adding drop-check rules for `FingerTree
        FingerTree::Single(1);
}
