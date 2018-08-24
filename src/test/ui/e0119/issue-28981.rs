use std::ops::Deref;

struct Foo;

impl<Foo> Deref for Foo { } //~ ERROR must be used
//~^ ERROR conflicting implementations

fn main() {}
