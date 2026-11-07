//@ revisions: e2018 e2021
//@[e2018] edition: 2018
//@[e2021] edition: 2021

// Parentheses around a pattern handed to a `macro_rules!` metavariable can be required by the
// matcher even though they are redundant in the expanded pattern, so removing them breaks the
// macro call. rust-lang/rust#86959

#![deny(unused_parens)]

macro_rules! match_pat {
    ($p:pat) => {
        match 'a' {
            $p => {}
            _ => {}
        }
    };
}

macro_rules! match_pat_param {
    ($p:pat_param) => {
        match 'a' {
            $p => {}
            _ => {}
        }
    };
}

macro_rules! if_let_pat {
    ($p:pat) => {
        if let $p = 'a' {}
    };
}

macro_rules! nested_pat {
    ($p:pat) => {
        match Some(1) {
            Some($p) => {}
            _ => {}
        }
    };
}

// `body_or_pat!()` breaks the context leg. its a `PatKind::Or`, but the parens and the arm
// both come from the macro body so they share one `SyntaxContext`, `eq_ctxt` is true and
// `avoid_or` is false. these are the macro authors own parens, no metavariable involved, so
// removing them is safe. this is the case that matters most: `from_expansion()` is true here
// and `!eq_ctxt` is false, so if the gate were ever keyed on "are we inside a macro" instead
// of "did this cross a context boundary" it would be wrongly silenced. nothing else in the
// file detects that.
macro_rules! body_or_pat {
    () => {
        match 'a' {
            ('a' | 'A') => {} //~ ERROR unnecessary parentheses around pattern
            _ => {}
        }
    };
}

fn main() {
    // `$p:pat` only accepts a top-level `|` from edition 2021 onwards.
    match_pat!(('a' | 'A'));
    // `$p:pat_param` never accepts a top-level `|`, in any edition.
    match_pat_param!(('a' | 'A'));
    if_let_pat!(('a' | 'A'));
    nested_pat!((1 | 2));
    // A leading `..=` is rejected by both fragment specifiers. rust-lang/rust#120737
    match_pat!((..='a' | 'z'));

    // `match_pat!(('a'))` breaks the or leg. `avoid_or` is true, since the pattern arrives
    // through `$p:pat` and its `SyntaxContext` differs from the arms. but the inner pattern is
    // `PatKind::Expr`, not `PatKind::Or`, so `unused.rs:676` never matches and the lint fires.
    // the suggestion rewrites the call site to `match_pat!('a')`, which is right — `$p:pat`
    // takes a bare literal. this guards against widening the gate to all patterns, or moving
    // the `avoid_or` check above the `match inner.kind`.
    match_pat!(('a')); //~ ERROR unnecessary parentheses around pattern
    body_or_pat!();
    // `let (_x) = 1;` is the no-macro baseline for the `check_expr` call site, one of the four
    // that went from a hardcoded `false` to a computed value.
    let (_x) = 1; //~ ERROR unnecessary parentheses around pattern
}
