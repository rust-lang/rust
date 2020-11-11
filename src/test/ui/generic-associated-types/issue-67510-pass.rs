#![feature(generic_associated_types)]

trait X {
    type Y<'a>;
}

fn func1<'a>(x: Box<dyn X<Y<'a>=&'a ()>>) {}

fn func2(x: Box<dyn for<'a> X<Y<'a>=&'a ()>>) {}

fn main() {}
