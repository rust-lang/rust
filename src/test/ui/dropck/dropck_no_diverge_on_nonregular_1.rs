// Issue #22443: reject code using non-regular types that would
// otherwise cause dropck to loop infinitely.

use std::marker::PhantomData;

struct Digit<T> {
    elem: T
}

struct Node<T:'static> { m: PhantomData<&'static T> }


enum FingerTree<T:'static> {
    Single(T),
    // Bug report indicated `Digit` after `Box` would stack-overflow (versus
    // `Digit` before `Box`; see `dropck_no_diverge_on_nonregular_2`).
    Deep(
        Box<FingerTree<Node<T>>>,
        Digit<T>,
        )
}

fn main() {
    let ft = FingerTree::Single(1);
    //~^ ERROR overflow while adding drop-check rules for `FingerTree<i32>` [E0320]
    //~^^ ERROR overflow while adding drop-check rules for `FingerTree<i32>` [E0320]
}
