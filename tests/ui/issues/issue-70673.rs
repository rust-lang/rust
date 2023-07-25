// Regression test for https://github.com/rust-lang/rust/issues/70673.

//@run

#![feature(thread_local)]

#[thread_local]
static A: &u8 = &42;

fn main() {
    dbg!(*A);
}
