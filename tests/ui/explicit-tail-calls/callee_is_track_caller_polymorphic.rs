//@ run-pass
//@ ignore-pass
//@ ignore-backends: gcc
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn c<T: Trait>() {
    become T::f();
    //~^ warning: tail calling a function marked with `#[track_caller]` has no special effect
}

trait Trait {
    #[track_caller]
    fn f() {}
}

impl Trait for () {}

fn main() {
    c::<()>();
}
