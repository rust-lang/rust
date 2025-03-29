// See #130494

#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

fn f(x: &pin  const i32) {}
fn g<'a>(x: &  'a pin const  i32) {}
fn h<'a>(x: &  'a pin  
mut i32) {}
fn i(x: &pin      mut  i32) {}

struct Foo;

impl Foo {
    fn f(&pin   const self) {}
    fn g<'a>(&   'a pin const    self) {}
    fn h<'a>(&    'a pin
mut self) {}
    fn i(&pin      mut   self) {}
}

fn borrows() {
    let mut foo = 0_i32;
    let x: Pin<&mut _> = & pin 
    mut    foo;

    let x: Pin<&_> = &
    pin                const 
    foo;
}
