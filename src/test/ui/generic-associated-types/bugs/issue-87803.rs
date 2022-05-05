// check-fail
// known-bug

// This should pass, but using a type alias vs a reference directly
// changes late-bound -> early-bound.

#![feature(generic_associated_types)]

trait Scanner {
    type Input<'a>;
    type Token<'a>;

    fn scan<'a>(&mut self, i : Self::Input<'a>) -> Self::Token<'a>;
}

struct IdScanner();

impl Scanner for IdScanner {
    type Input<'a> = &'a str;
    type Token<'a> = &'a str;

    fn scan<'a>(&mut self, s : &'a str) -> &'a str {
        s
    }
}

fn main() {}
