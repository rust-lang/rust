// skip-filecheck
// Regression test for issue #116212.

use std::mem::MaybeUninit;

struct Foo {
    x: u8,
    y: !,
}

fn main() {
    let foo = unsafe { MaybeUninit::<Foo>::uninit().assume_init() };
}
