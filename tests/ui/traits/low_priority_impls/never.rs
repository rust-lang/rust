// This test checks that low priority impls take precedence over never type fallback
//
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//
//@ check-pass

#![feature(rustc_attrs)]

#[derive(Default)]
struct X;
#[derive(Default)]
struct Y;

trait Meow<T> {
    fn f(x: T) -> Self;
}

impl<T> Meow<T> for T {
    fn f(x: T) -> T {
        x
    }
}

#[rustc_low_priority_impl]
impl Meow<Y> for X {
    fn f(Y: Y) -> X {
        X
    }
}

fn main() {
    let _: X = Meow::f(loop {});
    //~^ warn: dependency on trait impl fallback [trait_impl_fallback]
    //~| warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
