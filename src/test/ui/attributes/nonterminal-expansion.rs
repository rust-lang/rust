// Macros were previously expanded in `Expr` nonterminal tokens, now they are not.

macro_rules! pass_nonterminal {
    ($n:expr) => {
        #[repr(align($n))]
        //~^ ERROR unexpected token: `n`
        //~| ERROR incorrect `repr(align)` attribute format
        //~| ERROR incorrect `repr(align)` attribute format
        struct S;
    };
}

macro_rules! n {
    () => { 32 };
}

pass_nonterminal!(n!());

fn main() {}

// njn: petrochenkov: Looks like the parsing is somehow run twice when we are
// attempting to parse align($n) into a MetaItem? That shouldn't happen.`
