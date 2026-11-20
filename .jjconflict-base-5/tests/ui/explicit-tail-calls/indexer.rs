//@ run-pass
//@ ignore-backends: gcc
// Indexing taken from
// https://github.com/phi-go/rfcs/blob/guaranteed-tco/text%2F0000-explicit-tail-calls.md#tail-call-elimination
// no other test has utilized the "function table"
// described in the RFC aside from this one at this point.
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn f0(_: usize) {}
fn f1(_: usize) {}
fn f2(_: usize) {}

fn indexer(idx: usize) {
    let v: [fn(usize); 3] = [f0, f1, f2];
    become v[idx](idx)
}

fn main() {
    for idx in 0..3 {
        indexer(idx);
    }
}
