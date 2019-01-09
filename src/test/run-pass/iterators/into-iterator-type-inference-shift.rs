// run-pass
#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
// Regression test for type inference failure around shifting. In this
// case, the iteration yields an isize, but we hadn't run the full type
// propagation yet, and so we just saw a type variable, yielding an
// error.

// pretty-expanded FIXME #23616

trait IntoIterator {
    type Iter: Iterator;

    fn into_iter(self) -> Self::Iter;
}

impl<I> IntoIterator for I where I: Iterator {
    type Iter = I;

    fn into_iter(self) -> I {
        self
    }
}

fn desugared_for_loop_bad(byte: u8) -> u8 {
    let mut result = 0;
    let mut x = IntoIterator::into_iter(0..8);
    let mut y = Iterator::next(&mut x);
    let mut z = y.unwrap();
    byte >> z;
    1
}

fn main() {}
