// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

use std::ops::AddAssign;

fn inc<T: AddAssign + Clone, const N: usize>(v: &mut [T; N]) -> &mut [T; N] {
    for x in v.iter_mut() {
        *x += x.clone();
    }
    v
}

fn main() {
    let mut v = [1, 2, 3];
    inc(&mut v);
    assert_eq!(v, [2, 4, 6]);
}
