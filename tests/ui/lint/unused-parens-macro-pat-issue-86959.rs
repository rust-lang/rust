//@ check-pass
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

fn main() {
    // `$p:pat` only accepts a top-level `|` from edition 2021 onwards.
    match_pat!(('a' | 'A'));
    // `$p:pat_param` never accepts a top-level `|`, in any edition.
    match_pat_param!(('a' | 'A'));
    if_let_pat!(('a' | 'A'));
    nested_pat!((1 | 2));
    // A leading `..=` is rejected by both fragment specifiers. rust-lang/rust#120737
    match_pat!((..='a' | 'z'));
}
