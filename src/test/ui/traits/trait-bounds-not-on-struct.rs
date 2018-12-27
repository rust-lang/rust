#![allow(bare_trait_objects)]

struct Foo;

fn foo(_x: Box<Foo + Send>) { } //~ ERROR expected trait, found struct `Foo`

type A<T> = Box<dyn Vec<T>>; //~ ERROR expected trait, found struct `Vec`

fn main() { }
