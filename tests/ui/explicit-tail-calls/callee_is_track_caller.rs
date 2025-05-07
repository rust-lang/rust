//@ check-pass
// FIXME(explicit_tail_calls): make this run-pass, once tail calls are properly implemented
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn a(x: u32) -> u32 {
    become b(x);
}

#[track_caller]
fn b(x: u32) -> u32 { x + 42 }

fn main() {
    assert_eq!(a(12), 54);
}
