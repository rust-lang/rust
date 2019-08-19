// Various tests related to testing how region inference works
// with respect to the object receivers.

// build-pass (FIXME(62277): could be check-pass?)
#![allow(warnings)]

trait Foo {
    fn borrowed<'a>(&'a self) -> &'a ();
}

// Here the receiver and return value all have the same lifetime,
// so no error results.
fn borrowed_receiver_same_lifetime<'a>(x: &'a Foo) -> &'a () {
    x.borrowed()
}


fn main() {}
