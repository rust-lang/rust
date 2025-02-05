#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

#[track_caller]
fn a() {
    become b(); //~ error: a function marked with `#[track_caller]` cannot perform a tail-call
}

fn b() {}

#[track_caller]
fn c() {
    become a(); //~ error: a function marked with `#[track_caller]` cannot perform a tail-call
}

fn main() {}
