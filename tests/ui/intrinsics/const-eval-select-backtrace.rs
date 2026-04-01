#![feature(core_intrinsics)]
// See issue #100696.
//@ run-fail
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=0

#[track_caller]
fn uhoh() {
    panic!("Aaah!")
}

const fn c() {}

fn main() {
    std::intrinsics::const_eval_select((), c, uhoh);
}
