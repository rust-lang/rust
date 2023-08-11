// run-pass

// njn: this test now passes! `Interpolate` removal lifts the following restriction:
//
//   = note: captured metavariables except for `:tt`, `:ident` and `:lifetime` cannot be compared
//     to other tokens
//   = note: see
//     <doc.rust-lang.org/nightly/reference/macros-by-example.html#forwarding-a-matched-fragment>
//     for more information
//   = help: try using `:tt` instead in the macro definition

// Check that we are refusing to match on complex nonterminals for which tokens are
// unavailable and we'd have to go through AST comparisons.

#![feature(decl_macro)]

macro simple_nonterminal($nt_ident: ident, $nt_lifetime: lifetime, $nt_tt: tt) {
    macro n(a $nt_ident b $nt_lifetime c $nt_tt d) {
        struct _S;
    }

    n!(a $nt_ident b $nt_lifetime c $nt_tt d);
}

macro complex_nonterminal($nt_item: item) {
    macro n(a $nt_item b) {
        struct _S;
    }

    n!(a $nt_item b);
}

simple_nonterminal!(a, 'a, (x, y, z)); // OK

complex_nonterminal!(enum E {});

fn main() {}
