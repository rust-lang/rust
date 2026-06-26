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

    n!(a $nt_item b); //~ ERROR no rules expected `item` metavariable
}

simple_nonterminal!(a, 'a, (x, y, z)); // OK

complex_nonterminal!(enum E {});

// `ident`, `lifetime`, and `tt` all work. Other fragments do not. See
// https://doc.rust-lang.org/nightly/reference/macros-by-example.html#forwarding-a-matched-fragment
macro_rules! foo {
    (ident $x:ident) => { bar!(ident $x); };
    (lifetime $x:lifetime) => { bar!(lifetime $x); };
    (tt $x:tt) => { bar!(tt $x); };
    (expr $x:expr) => { bar!(expr $x); }; //~ ERROR: no rules expected `expr` metavariable
    (literal $x:literal) => { bar!(literal $x); }; //~ ERROR: no rules expected `literal` metavariable
    (path $x:path) => { bar!(path $x); }; //~ ERROR: no rules expected `path` metavariable
    (stmt $x:stmt) => { bar!(stmt $x); }; //~ ERROR: no rules expected `stmt` metavariable
}

macro_rules! bar {
    (ident abc) => {};
    (lifetime 'abc) => {};
    (tt 2) => {};
    (expr 3) => {};
    (literal 4) => {};
    (path a::b::c) => {};
    (stmt let abc = 0) => {};
}

foo!(ident abc);
foo!(lifetime 'abc);
foo!(tt 2);
foo!(expr 3);
foo!(literal 4);
foo!(path a::b::c);
foo!(stmt let abc = 0);

fn main() {}
