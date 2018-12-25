// run-pass
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![feature(box_syntax)]

use std::io::{self, Write};

trait Trait {
    fn f(&self);
}

#[derive(Copy, Clone)]
struct Struct {
    x: isize,
    y: isize,
}

impl Trait for Struct {
    fn f(&self) {
        println!("Hi!");
    }
}

fn foo(mut a: Box<Write>) {}

pub fn main() {
    let a = Struct { x: 1, y: 2 };
    let b: Box<Trait> = Box::new(a);
    b.f();
    let c: &Trait = &a;
    c.f();

    let out = io::stdout();
    foo(Box::new(out));
}
