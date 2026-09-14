//! Regression test for <https://github.com/rust-lang/rust/issues/41479>.
//! Field access on zst tuple items used to trigger LLVM assertion.
//@ run-pass

fn split<A, B>(pair: (A, B)) {
    let _a = pair.0;
    let _b = pair.1;
}

fn main() {
    split(((), ((), ())));
}
