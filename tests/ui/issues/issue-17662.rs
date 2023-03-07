// run-pass
// aux-build:issue-17662.rs


extern crate issue_17662 as i;

use std::marker;

struct Bar<'a> { m: marker::PhantomData<&'a ()> }

impl<'a> i::Foo<'a, usize> for Bar<'a> {
    fn foo(&self) -> usize { 5 }
}

pub fn main() {
    assert_eq!(i::foo(&Bar { m: marker::PhantomData }), 5);
}
