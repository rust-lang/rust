// revisions: normal opt
// check-pass
//[opt] compile-flags: -Zmir-opt-level=3

#![feature(unboxed_closures)]

extern "rust-call" fn foo<T>(_: T) {}

fn main() {
    foo(());
    foo((1, 2));
}
