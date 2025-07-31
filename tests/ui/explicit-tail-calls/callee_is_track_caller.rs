#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn a() {
    become b(); //~ error: a function marked with `#[track_caller]` cannot be tail-called
}

#[track_caller]
fn b() {}

fn main() {}
