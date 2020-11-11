// Only accept lifetimes or types as generic arguments for associated types 
// in trait paths

#![feature(generic_associated_types)]

trait X { 
    type Y<'a>;
}

fn f1<'a>(arg : Box<dyn X<Y<1> = &'a ()>>) {}  
    //~^  ERROR: generic arguments of associated types must be lifetimes or types
    //~^^ ERROR: wrong number of lifetime arguments: expected 1, found 0 

fn main() {}
