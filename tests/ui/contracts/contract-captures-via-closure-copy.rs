//@ run-crash
//@ compile-flags: -Zcontract-checks=yes

#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

struct Baz {
    baz: i32
}

#[track_caller]
#[core::contracts::requires(x.baz > 0)]
#[core::contracts::ensures({let old = x.baz; move |ret:&Baz| ret.baz == old*2 })]
// Relevant thing is this:  ^^^^^^^^^^^^^^^
// because we are capturing state that is Copy
fn doubler(x: Baz) -> Baz {
    Baz { baz: x.baz + 10 }
}

fn main() {
    assert_eq!(doubler(Baz { baz: 10 }).baz, 20);
    assert_eq!(doubler(Baz { baz: 100 }).baz, 200);
    // This is a *run-fail* test because it is still exercising the
    // contract machinery, specifically because this second invocation
    // of `doubler` shows how the code does not meet its contract.
}
