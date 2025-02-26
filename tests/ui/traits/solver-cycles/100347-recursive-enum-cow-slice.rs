//@ check-pass

use std::borrow::Cow;

#[derive(Clone)]
enum Test<'a> {
    Int(u8),
    Array(Cow<'a, [Test<'a>]>),
}

fn main() {}
