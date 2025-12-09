//@ compile-flags: -Zdeduplicate-diagnostics=yes

// Macros were previously expanded in `Expr` nonterminal tokens, now they are not.

macro_rules! pass_nonterminal {
    ($n:expr) => {
        #[repr(align($n))]
        fn foo() {}
    };
}

macro_rules! n {
    () => { 32 };
}

pass_nonterminal!(n!());
//~^ ERROR expected one of `(`, `::`, or `=`, found `!`

fn main() {}
