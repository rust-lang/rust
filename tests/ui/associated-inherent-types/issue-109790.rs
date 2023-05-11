// check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo<T>(T);

impl<'a> Foo<fn(&'a ())> {
    type Assoc = &'a ();
}

trait Other {}
impl Other for u32 {}

fn bar(_: for<'a> fn(Foo<fn(&'a ())>::Assoc)) {}

fn main() {}
