// run-pass
// revisions: default mir-opt
//[mir-opt] compile-flags: -Zmir-opt-level=3

#![feature(track_caller)]

#[track_caller]
fn f() {}

fn main() {
    f();
}
