//@ edition:2015..2021
//@ compile-flags: -Zcontract-checks=yes

#![expect(incomplete_features)]
#![feature(contracts)]

struct Baz {
    baz: i32
}

#[track_caller]
#[core::contracts::requires(x.baz > 0)]
#[core::contracts::ensures({let old = x; move |ret:&Baz| ret.baz == old.baz*2 })]
// Relevant thing is this:  ^^^^^^^^^^^
// because we are capturing state that is non-Copy.
//~^^^ ERROR trait bound `Baz: std::marker::Copy` is not satisfied
fn doubler(x: Baz) -> Baz {
    Baz { baz: x.baz + 10 }
}

fn main() {
    assert_eq!(doubler(Baz { baz: 10 }).baz, 20);
    assert_eq!(doubler(Baz { baz: 100 }).baz, 200);
}
