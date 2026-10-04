#![feature(unboxed_closures)]

trait Three<A,B,C> { fn dummy(&self) -> (A,B,C); }

fn foo(_: &dyn Three())
//~^ ERROR trait takes 3 generic arguments but 1 generic argument
//~| ERROR associated type `Output` not found
{}

fn main() { }
