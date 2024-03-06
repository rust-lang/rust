//@ run-pass
//@ compile-flags:-Zmir-opt-level=3

// Previously ICEd because we did not normalize during inlining,
// see https://github.com/rust-lang/rust/pull/77306 for more discussion.

pub fn write() {
    create()()
}

pub fn create() -> impl FnOnce() {
   || ()
}

fn main() {
    write();
}
