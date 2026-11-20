//! Regression test for https://github.com/rust-lang/rust/issues/15034

pub struct Lexer<'a> {
    input: &'a str,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Lexer<'a> {
        Lexer { input: input }
    }
}

struct Parser<'a> {
    lexer: &'a mut Lexer<'a>,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: &'a mut Lexer) -> Parser<'a> {
        Parser { lexer: lexer }
        //~^ ERROR explicit lifetime required in the type of `lexer` [E0621]
    }
}

fn main() {}
