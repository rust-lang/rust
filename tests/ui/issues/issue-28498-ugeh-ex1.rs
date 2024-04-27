//@ run-pass

// Example taken from RFC 1238 text

// https://github.com/rust-lang/rfcs/blob/master/text/1238-nonparametric-dropck.md
//     #example-of-the-unguarded-escape-hatch

#![feature(dropck_eyepatch)]
use std::cell::Cell;

struct Concrete<'a>(#[allow(dead_code)] u32, Cell<Option<&'a Concrete<'a>>>);

struct Foo<T> { data: Vec<T> }

// Below is the UGEH attribute
unsafe impl<#[may_dangle] T> Drop for Foo<T> {
    fn drop(&mut self) { }
}

fn main() {
    let mut foo = Foo {  data: Vec::new() };
    foo.data.push(Concrete(0, Cell::new(None)));
    foo.data.push(Concrete(0, Cell::new(None)));

    foo.data[0].1.set(Some(&foo.data[1]));
    foo.data[1].1.set(Some(&foo.data[0]));
}
