// check-pass

#![feature(associated_type_bounds)]

trait A<'a> {
    type Assoc: ?Sized;
}

impl<'a> A<'a> for () {
    type Assoc = &'a ();
}

fn hello() -> impl for<'a> A<'a, Assoc: Sized> {
    ()
}

fn main() {}
