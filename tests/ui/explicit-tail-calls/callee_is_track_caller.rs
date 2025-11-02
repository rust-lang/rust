//@ run-pass
//@ ignore-pass
//@ ignore-backends: gcc
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn a(x: u32) -> u32 {
    become b(x);
    //~^ warning: tail calling a function marked with `#[track_caller]` has no special effect
}

#[track_caller]
fn b(x: u32) -> u32 { x + 42 }

fn main() {
    assert_eq!(a(12), 54);
}
