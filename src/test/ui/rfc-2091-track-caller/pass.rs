// run-pass
#![feature(track_caller)]

#[track_caller]
fn f() {}

fn main() {
    f();
}
