#![deny(clippy::while_let_on_iterator)]
#![allow(unused_mut)]
#![allow(clippy::uninlined_format_args)]

use std::iter::Iterator;

struct Foo;

impl Foo {
    fn foo1<I: Iterator<Item = usize>>(mut it: I) {
        while let Some(_) = it.next() {
            println!("{:?}", it.size_hint());
        }
    }

    fn foo2<I: Iterator<Item = usize>>(mut it: I) {
        while let Some(e) = it.next() {
            println!("{:?}", e);
        }
    }
}

fn main() {
    Foo::foo1(vec![].into_iter());
    Foo::foo2(vec![].into_iter());
}
