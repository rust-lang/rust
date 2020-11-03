// Macros were previously expanded in `Expr` nonterminal tokens, now they are not.

macro_rules! pass_nonterminal {
    ($n:expr) => {
        #[repr(align($n))] //~ ERROR expected unsuffixed literal or identifier, found `n!()`
                           //~| ERROR unrecognized representation hint
        struct S;
    };
}

macro_rules! n {
    () => { 32 };
}

pass_nonterminal!(n!());

fn main() {}
