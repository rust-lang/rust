//@ check-pass

#![feature(type_changing_struct_update)]

use std::any::Any;

struct Foo<A, B: ?Sized, C: ?Sized> {
    a: A,
    b: Box<B>,
    c: Box<C>,
}

struct B;
struct C;

fn main() {
    let y = Foo::<usize, dyn Any, dyn Any> {
        a: 0,
        b: Box::new(B),
        ..Foo {
            a: 0,
            b: Box::new(B),
            // C needs to be told to coerce to `Box<dyn Any>`
            c: Box::new(C),
        }
    };
}
