//@ build-fail
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn c<T: Trait>() {
    // `T::f` is not known when checking tail calls,
    // so this has to be checked by the backend.
    become T::f();
}

trait Trait {
    #[track_caller]
    // FIXME(explicit_tail_calls): make error point to the tail call, not callee definition
    fn f() {} //~ error: a function marked with `#[track_caller]` cannot be tail-called
}

impl Trait for () {}

fn main() {
    c::<()>();
}
