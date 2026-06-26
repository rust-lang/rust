// Tests that using fulfillment in the trait solver means that we detect that a
// method is impossible, leading to no ambiguity.
//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

struct W<T, U>(Option<T>, Option<U>);

impl<'a> W<fn(&'a ()), u32> {
    fn method(&self) {}
}

trait Leak {}
impl<T: Fn(&())> Leak for T {}

impl<T: Leak> W<T, i32> {
    fn method(&self) {}
}

fn test<'a>() {
    let x: W<fn(&'a ()), _> = W(None, None);
    x.method();
}

fn main() {}
