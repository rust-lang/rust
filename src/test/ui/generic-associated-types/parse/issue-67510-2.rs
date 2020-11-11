// Only accept generic arguments of typeTyKind::Path as associated types in
// trait paths
#![feature(generic_associated_types)]

trait X { 
    type Y<'a>;
}

fn f1<'a>(arg : Box<dyn X<(Y<'a>) = &'a ()>>) {} 
    //~^ ERROR: Expected the name of an associated type

fn main() {}
