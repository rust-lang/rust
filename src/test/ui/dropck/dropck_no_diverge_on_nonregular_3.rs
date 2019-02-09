// Issue #22443: reject code using non-regular types that would
// otherwise cause dropck to loop infinitely.
//
// This version is just checking that we still sanely handle a trivial
// wrapper around the non-regular type. (It also demonstrates how the
// error messages will report different types depending on which type
// dropck is analyzing.)

// ignore-tidy-linelength

use std::marker::PhantomData;

struct Digit<T> {
    elem: T
}

struct Node<T:'static> { m: PhantomData<&'static T> }

enum FingerTree<T:'static> {
    Single(T),
    // Bug report indicated `Digit` before `Box` would loop infinitely.
    Deep(
        Digit<T>,
        Box<FingerTree<Node<T>>>,
        )
}

enum Wrapper<T:'static> {
    Simple,
    Other(FingerTree<T>),
}

fn main() {
    let w = Some(Wrapper::Simple::<u32>);
    //~^ ERROR overflow while adding drop-check rules for `std::option::Option<Wrapper<u32>>` [E0320]
    //~^^ ERROR overflow while adding drop-check rules for `std::option::Option<Wrapper<u32>>` [E0320]
    //~| ERROR overflow while adding drop-check rules for `Wrapper<u32>` [E0320]
}
