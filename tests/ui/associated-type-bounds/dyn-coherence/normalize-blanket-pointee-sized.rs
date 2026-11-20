//@ check-pass

// Determining that `Sub` is dyn-compatible requires normalizing `<Self as Constant>::Output`.
// This depends on impl blocks elsewhere. Make sure we can do this.
// Due to the `PointeeSized`, this normalization doesn't need any trait bounds on `Self`.

#![feature(sized_hierarchy)]

use std::marker::PointeeSized;

trait Constant: PointeeSized {
    type Output;
}
impl<T: PointeeSized> Constant for T {
    type Output = i32;
}

trait Super {
    type Assoc;
}

trait Sub: Super<Assoc = <Self as Constant>::Output> + Super<Assoc = i32> {}

fn foo(_: &dyn Sub) {}
fn main() {}
