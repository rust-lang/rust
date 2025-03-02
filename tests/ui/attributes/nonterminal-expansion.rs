//@ compile-flags: -Zdeduplicate-diagnostics=yes

// Macros were previously expanded in `Expr` nonterminal tokens, now they are not.

macro_rules! pass_nonterminal {
    ($n:expr) => {
        #[repr(align($n))]
        //~^ ERROR expected unsuffixed literal, found expression `n!()`
        //~^^ ERROR incorrect `repr(align)` attribute format: `align` expects a literal integer as argument [E0693]
        struct S;
    };
}

macro_rules! n {
    () => { 32 };
}

pass_nonterminal!(n!());

fn main() {}
