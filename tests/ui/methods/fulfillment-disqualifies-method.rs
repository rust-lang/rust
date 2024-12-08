// Tests that using fulfillment in the trait solver means that we detect that a
// method is impossible, leading to no ambiguity.
//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#[derive(Default)]
struct W<A, B>(A, B);

trait Constrain {
    type Output;
}

impl Constrain for i32 {
    type Output = u32;
}

trait Impossible {}

impl<A, B> W<A, B> where A: Constrain<Output = B>, B: Impossible {
    fn method(&self) {}
}

impl W<i32, u32> {
    fn method(&self) {}
}

fn main() {
    let w: W<i32, _> = W::default();
    w.method();
}
