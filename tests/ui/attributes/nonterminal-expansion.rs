//@ compile-flags: -Zdeduplicate-diagnostics=yes

// Macros were previously expanded in `Expr` nonterminal tokens, now they are not.

macro_rules! pass_nonterminal {
    ($n:expr) => {
        #[repr(align($n))]
        //~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `expr` metavariable
        struct S;
    };
}

macro_rules! n {
    () => { 32 };
}

pass_nonterminal!(n!());

fn main() {}
