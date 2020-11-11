// Parser shouldn't accept parenthesized generic arguments for associated types in
// trait paths
#![feature(generic_associated_types)]

trait X { 
    type Y<'a>;
}

fn f1<'a>(arg : Box<dyn X<Y('a) = &'a ()>>) {}  
    //~^  ERROR: lifetime in trait object type must be followed by `+`
    //~^^ ERROR: invalid use of parenthesized generic arguments, expected angle 
    //           bracketed (`<...>`) generic arguments
    //~^^^^ ERROR: wrong number of lifetime arguments: expected 1, found 0
    

fn main() {}
