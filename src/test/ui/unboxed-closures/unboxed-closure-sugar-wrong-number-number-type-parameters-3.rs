#![feature(unboxed_closures)]

trait Three<A,B,C> { fn dummy(&self) -> (A,B,C); }

fn foo(_: &dyn Three())
//~^ ERROR this trait takes 3 type arguments but only 1 type argument was supplied
//~| ERROR associated type `Output` not found
{}

fn main() { }
