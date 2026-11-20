// Regression test of #85581.
// Checks not to suggest to add `;` when the second mutable borrow
// is in the first's scope.

use std::collections::BinaryHeap;

fn foo(heap: &mut BinaryHeap<i32>) {
    match heap.peek_mut() {
        Some(_) => { heap.pop(); },
        //~^ ERROR: cannot borrow `*heap` as mutable more than once at a time
        None => (),
    }
}

fn main() {}
