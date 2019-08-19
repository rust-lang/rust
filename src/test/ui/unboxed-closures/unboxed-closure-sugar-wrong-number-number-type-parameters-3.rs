#![feature(unboxed_closures)]

trait Three<A,B,C> { fn dummy(&self) -> (A,B,C); }

fn foo(_: &dyn Three())
//~^ ERROR wrong number of type arguments
//~| ERROR associated type `Output` not found
{}

fn main() { }
