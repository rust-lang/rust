// error-pattern: reached the recursion limit while auto-dereferencing

#![feature(box_syntax)]

use std::ops::Deref;

struct Foo;

impl Deref for Foo {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        self
    }
}

pub fn main() {
    let mut x;
    loop {
        x = box x;
        x.foo;
        x.bar();
    }

    Foo.foo;
    Foo.bar();
}
