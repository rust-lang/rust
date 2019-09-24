// run-pass

use std::cell::Cell;
use std::ops::{Deref, DerefMut};
use std::vec::Vec;

struct DerefCounter<T> {
    count_imm: Cell<usize>,
    count_mut: usize,
    value: T
}

impl<T> DerefCounter<T> {
    fn new(value: T) -> DerefCounter<T> {
        DerefCounter {
            count_imm: Cell::new(0),
            count_mut: 0,
            value: value
        }
    }

    fn counts(&self) -> (usize, usize) {
        (self.count_imm.get(), self.count_mut)
    }
}

impl<T> Deref for DerefCounter<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.count_imm.set(self.count_imm.get() + 1);
        &self.value
    }
}

impl<T> DerefMut for DerefCounter<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.count_mut += 1;
        &mut self.value
    }
}

pub fn main() {
    let mut n = DerefCounter::new(0);
    let mut v = DerefCounter::new(Vec::new());

    let _ = *n; // Immutable deref + copy a POD.
    assert_eq!(n.counts(), (1, 0));

    let _ = (&*n, &*v); // Immutable deref + borrow.
    assert_eq!(n.counts(), (2, 0)); assert_eq!(v.counts(), (1, 0));

    let _ = (&mut *n, &mut *v); // Mutable deref + mutable borrow.
    assert_eq!(n.counts(), (2, 1)); assert_eq!(v.counts(), (1, 1));

    let mut v2 = Vec::new();
    v2.push(1);

    *n = 5; *v = v2; // Mutable deref + assignment.
    assert_eq!(n.counts(), (2, 2)); assert_eq!(v.counts(), (1, 2));

    *n -= 3; // Mutable deref + assignment with binary operation.
    assert_eq!(n.counts(), (2, 3));

    // Immutable deref used for calling a method taking &self. (The
    // typechecker is smarter now about doing this.)
    (*n).to_string();
    assert_eq!(n.counts(), (3, 3));

    // Mutable deref used for calling a method taking &mut self.
    (*v).push(2);
    assert_eq!(v.counts(), (1, 3));

    // Check the final states.
    assert_eq!(*n, 2);
    let expected: &[_] = &[1, 2];
    assert_eq!((*v), expected);
}
