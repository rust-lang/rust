#![feature(unboxed_closures)]

trait Zero { fn dummy(&self); }

fn foo(_: dyn Zero())
    //~^ ERROR wrong number of type arguments
    //~| ERROR associated type `Output` not found for `Zero`
{}

fn main() { }
