//@ run-pass

use std::cell::Cell;

const NONE_CELL_STRING: Option<Cell<String>> = None;

struct Foo<T>(#[allow(dead_code)] T);
impl<T> Foo<T> {
    const FOO: Option<Box<T>> = None;
}

fn main() {
    let _: &'static u32 = &42;
    let _: &'static Option<u32> = &None;

    // We should be able to peek at consts and see they're None.
    let _: &'static Option<Cell<String>> = &NONE_CELL_STRING;
    let _: &'static Option<Box<()>> = &Foo::FOO;
}
