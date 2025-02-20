#![allow(unused)]
#![warn(clippy::sliced_string_as_bytes)]

use std::ops::{Index, Range};

struct Foo;

struct Bar;

impl Bar {
    fn as_bytes(&self) -> &[u8] {
        &[0, 1, 2, 3]
    }
}

impl Index<Range<usize>> for Foo {
    type Output = Bar;

    fn index(&self, _: Range<usize>) -> &Self::Output {
        &Bar
    }
}

fn main() {
    let s = "Lorem ipsum";
    let string: String = "dolor sit amet".to_owned();

    let bytes = s[1..5].as_bytes();
    //~^ sliced_string_as_bytes
    let bytes = string[1..].as_bytes();
    //~^ sliced_string_as_bytes
    let bytes = "consectetur adipiscing"[..=5].as_bytes();
    //~^ sliced_string_as_bytes

    let f = Foo;
    let bytes = f[0..4].as_bytes();
}
