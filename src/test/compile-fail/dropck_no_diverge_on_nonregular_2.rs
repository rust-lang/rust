// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
    let ft = //~ ERROR overflow while adding drop-check rules for FingerTree
        FingerTree::Single(1);
    //~^ ERROR overflow while adding drop-check rules for FingerTree
}
