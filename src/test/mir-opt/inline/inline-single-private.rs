// ignore-wasm32-bare compiled with panic=abort by default
// compile-flags: -Zinline-mir-hint-threshold=50 -Zmir-opt-level=3

#[inline]
fn one_caller() {
    println!("Hellow world!");
}

#[inline]
fn more_than_one_call() {
    println!("Hellow world!");
}

fn caller1() {
    more_than_one_call();
    more_than_one_call();
}

// EMIT_MIR inline_single_private.main.Inline.diff
fn main() {
    one_caller();
    more_than_one_call();
}
