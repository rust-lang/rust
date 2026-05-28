// Regression test for issue #109299.

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Lexer<'d>(&'d ());

impl Lexer<'static> {
    type Cursor = ();
}

fn test(_: Lexer::Cursor) {} //~ ERROR mismatched types
//~^ ERROR: lifetime may not live long enough

fn main() {}
