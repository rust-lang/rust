// run-pass
#![feature(track_caller)] //~ WARN the feature `track_caller` is incomplete

#[track_caller]
fn f() {}

fn main() {
    f();
}
