//! Regression test for issue <https://github.com/rust-lang/rust/issues/158017>.
//@ run-pass
//@ compile-flags: -C opt-level=3
//@ ignore-backends: gcc
//@ ignore-wasm
//@ ignore-riscv64

#![feature(explicit_tail_calls, unboxed_closures)]
#![expect(incomplete_features)]

#[inline(never)]
fn seed_stack() {
    let mut values = [100_u64; 8];
    std::hint::black_box(&mut values);
}

#[inline(never)]
extern "rust-call" fn callee((value,): ([u64; 4],)) -> u64 {
    value[0]
}

#[inline(never)]
extern "rust-call" fn caller((_,): ([u64; 4],)) -> u64 {
    become callee(([5, 6, 7, 8],));
}

fn main() {
    seed_stack();
    assert_eq!(5, caller(([1, 2, 3, 4],)));
}
