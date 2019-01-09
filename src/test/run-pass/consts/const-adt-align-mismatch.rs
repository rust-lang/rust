// run-pass
#![allow(dead_code)]
#![allow(deprecated)]

use std::mem;

#[derive(PartialEq, Debug)]
enum Foo {
    A(u32),
    Bar([u16; 4]),
    C
}

// NOTE(eddyb) Don't make this a const, needs to be a static
// so it is always instantiated as a LLVM constant value.
static FOO: Foo = Foo::C;

fn main() {
    assert_eq!(FOO, Foo::C);
    assert_eq!(mem::size_of::<Foo>(), 12);
    assert_eq!(mem::min_align_of::<Foo>(), 4);
}
