#![feature(generic_associated_types)]

trait X {
    type Y<'a>;
}

fn f(x: Box<dyn X<Y<'a>=&'a ()>>) {}

fn main() {}
