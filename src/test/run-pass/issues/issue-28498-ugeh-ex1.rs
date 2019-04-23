// run-pass
#![allow(deprecated)] // FIXME: switch to `#[may_dangle]` below.

// Example taken from RFC 1238 text

// https://github.com/rust-lang/rfcs/blob/master/text/1238-nonparametric-dropck.md
//     #example-of-the-unguarded-escape-hatch

#![feature(dropck_parametricity)]
use std::cell::Cell;

struct Concrete<'a>(u32, Cell<Option<&'a Concrete<'a>>>);

struct Foo<T> { data: Vec<T> }

impl<T> Drop for Foo<T> {
    // Below is the UGEH attribute
    #[unsafe_destructor_blind_to_params]
    fn drop(&mut self) { }
}

fn main() {
    let mut foo = Foo {  data: Vec::new() };
    foo.data.push(Concrete(0, Cell::new(None)));
    foo.data.push(Concrete(0, Cell::new(None)));

    foo.data[0].1.set(Some(&foo.data[1]));
    foo.data[1].1.set(Some(&foo.data[0]));
}
