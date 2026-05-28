//@ run-pass
#![allow(unused_mut)]
#![allow(unused_variables)]
// Test that we choose Deref or DerefMut appropriately based on mutability of ref bindings (#15609).

use std::ops::{Deref, DerefMut};

struct DerefOk<T>(T);
struct DerefMutOk<T>(T);

impl<T> Deref for DerefOk<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for DerefOk<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        panic!()
    }
}

impl<T> Deref for DerefMutOk<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        panic!()
    }
}

impl<T> DerefMut for DerefMutOk<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn main() {
    // Check that mutable ref binding in match picks DerefMut
    let mut b = DerefMutOk(0);
    match *b {
        ref mut n => n,
    };

    // Check that mutable ref binding in let picks DerefMut
    let mut y = DerefMutOk(1);
    let ref mut z = *y;

    // Check that immutable ref binding in match picks Deref
    let mut b = DerefOk(2);
    match *b {
        ref n => n,
    };

    // Check that immutable ref binding in let picks Deref
    let mut y = DerefOk(3);
    let ref z = *y;

    // Check that mixed mutable/immutable ref binding in match picks DerefMut
    let mut b = DerefMutOk((0, 9));
    match *b {
        (ref mut n, ref m) => (n, m),
    };

    let mut b = DerefMutOk((0, 9));
    match *b {
        (ref n, ref mut m) => (n, m),
    };

    // Check that mixed mutable/immutable ref binding in let picks DerefMut
    let mut y = DerefMutOk((1, 8));
    let (ref mut z, ref a) = *y;

    let mut y = DerefMutOk((1, 8));
    let (ref z, ref mut a) = *y;

    // Check that multiple immutable ref bindings in match picks Deref
    let mut b = DerefOk((2, 7));
    match *b {
        (ref n, ref m) => (n, m),
    };

    // Check that multiple immutable ref bindings in let picks Deref
    let mut y = DerefOk((3, 6));
    let (ref z, ref a) = *y;

    // Check that multiple mutable ref bindings in match picks DerefMut
    let mut b = DerefMutOk((4, 5));
    match *b {
        (ref mut n, ref mut m) => (n, m),
    };

    // Check that multiple mutable ref bindings in let picks DerefMut
    let mut y = DerefMutOk((5, 4));
    let (ref mut z, ref mut a) = *y;
}
