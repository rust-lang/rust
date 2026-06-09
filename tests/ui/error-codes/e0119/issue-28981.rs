use std::ops::Deref;

struct Foo;

impl<Foo> Deref for Foo { } //~ ERROR must be used

fn main() {}
