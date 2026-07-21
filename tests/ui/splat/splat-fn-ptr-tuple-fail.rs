//! Test using `#[splat]` on tuple arguments of pointers to invalid simple functions.
//! Bug #158603 regression test
//@ run-fail
//@ check-run-results
//@ exec-env: RUST_BACKTRACE=0

//@ normalize-stderr: "thread '.*'" -> "thread 'NAME'"
//@ normalize-stderr: "note: run with.*\n" -> ""

#![expect(incomplete_features)]
#![feature(splat)]

fn main() {
    // FIXME(rustfmt): the attribute gets deleted by rustfmt
    #[rustfmt::skip]
    let x: fn(#[splat] (i32,)) = None.unwrap();
    x(1);
}
