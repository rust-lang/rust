#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Lexer<'d>(&'d ());

impl Lexer<'d> { //~ ERROR use of undeclared lifetime name `'d`
    type Cursor = ();
}

fn test(_: Lexer::Cursor) {}
//~^ ERROR: lifetime may not live long enough

fn main() {}
