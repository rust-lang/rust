// Test to ensure that only GenericArg::Type is accepted in associated type
// constraints
#![feature(generic_associated_types)]

trait X { 
    type Y<'a>;
}

fn f1<'a>(arg : Box<dyn X<'a = &'a ()>>) {}  
    //~^ ERROR: Expected the name of an associated type

fn main() {}
