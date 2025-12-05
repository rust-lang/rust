// regression test for <https://github.com/rust-lang/rust/issues/134336>
// this previously caused an ICE, because we would compare `#[track_caller]` of
// the callee and the caller (in tailcalls specifically), leading to a problem
// since `T::f`'s instance can't be resolved (we do not know if the function is
// or isn't marked with `#[track_caller]`!)
//
//@ check-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

trait Tr {
    fn f();
}

fn g<T: Tr>() {
    become T::f();
}

fn main() {}
