// Check that we are refusing to match on complex nonterminals for which tokens are
// unavailable and we'd have to go through AST comparisons.

#![feature(decl_macro)]

macro simple_nonterminal($nt_ident: ident, $nt_lifetime: lifetime, $nt_tt: tt) {
    macro n(a $nt_ident b $nt_lifetime c $nt_tt d) {
        struct S;
    }

    n!(a $nt_ident b $nt_lifetime c $nt_tt d);
}

macro complex_nonterminal($nt_item: item) {
    macro n(a $nt_item b) {
        struct S;
    }

    n!(a $nt_item b); //~ ERROR no rules expected the token `enum E { }`
}

simple_nonterminal!(a, 'a, (x, y, z)); // OK

complex_nonterminal!(enum E {});

fn main() {}
