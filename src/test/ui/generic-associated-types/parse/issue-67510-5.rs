// Test to ensure that the Constraint branch of the pattern match on AngleBracketedArg
// in `get_assoc_type_with_generic_args` is unreachable
#![feature(generic_associated_types)]

trait X { 
    type Y<'a>;
}

fn f1<'a>(arg : Box<dyn X<Y = B = &'a ()>>) {}  
    //~^ ERROR: Expected the name of an associated type

fn main() {}
