// In several type analysis passes we employ a specialized expansion procedure.
// This procedure used to incorrectly track expansion depth (growing much faster
// than normalization depth) resulting in its internal assertion triggering.
//
// issue: <https://github.com/rust-lang/rust/issues/142419>
//@ check-pass
#![feature(lazy_type_alias)]
#![expect(incomplete_features)]

type T0 = (T1, T1, T1, T1);
type T1 = (T2, T2, T2, T2);
type T2 = (T3, T3, T3, T3);
type T3 = (T4, T4, T4, T4);
type T4 = (T5, T5, T5, T5);
type T5 = (T6, T6, T6, T6);
type T6 = (T7, T7, T7, T7);
type T7 = ();

fn accept(_: T0) {}
fn main() {}
