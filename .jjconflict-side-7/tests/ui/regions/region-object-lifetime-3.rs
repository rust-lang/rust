// Various tests related to testing how region inference works
// with respect to the object receivers.

//@ check-pass
#![allow(warnings)]

trait Foo {
    fn borrowed<'a>(&'a self) -> &'a ();
}

// Borrowed receiver with two distinct lifetimes, but we know that
// 'b:'a, hence &'a () is permitted.
fn borrowed_receiver_related_lifetimes<'a,'b>(x: &'a (dyn Foo+'b)) -> &'a () {
    x.borrowed()
}


fn main() {}
