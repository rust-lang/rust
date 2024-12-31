//@ normalize-stderr: "\d+ bits" -> "N bits"

use std::mem;
use std::marker::PhantomData;

trait Foo {
    type Error;
}

struct Bar<U: Foo> {
    stream: PhantomData<U::Error>,
}

fn foo<U: Foo>(x: [usize; 2]) -> Bar<U> {
    unsafe { mem::transmute(x) }
    //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types
}

fn main() {}
