// check-pass

use std::ops::Range;

struct Foo;

impl From<<Range<usize> as Iterator>::Item> for Foo {
    fn from(_: <Range<usize> as Iterator>::Item) -> Foo {
        Foo
    }
}

fn main() {}
