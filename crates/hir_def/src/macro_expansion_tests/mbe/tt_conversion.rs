//! Unlike rustc, rust-analyzer's syntax tree are not "made of" token trees.
//! Rather, token trees are an explicit bridge between the parser and
//! (procedural or declarative) macros.
//!
//! This module tests tt <-> syntax tree conversion specifically
use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn round_trips_compound_tokens() {
    check(
        r#"
macro_rules! m {
    () => { type qual: ::T = qual::T; }
}
m!();
"#,
        expect![[r#"
macro_rules! m {
    () => { type qual: ::T = qual::T; }
}
type qual: ::T = qual::T;
"#]],
    )
}

#[test]
fn round_trips_literals() {
    check(
        r#"
macro_rules! m {
    () => {
        let _ = 'c';
        let _ = 1000;
        let _ = 12E+99_f64;
        let _ = "rust1";
        let _ = -92;
    }
}
fn f() {
    m!()
}
"#,
        expect![[r#"
macro_rules! m {
    () => {
        let _ = 'c';
        let _ = 1000;
        let _ = 12E+99_f64;
        let _ = "rust1";
        let _ = -92;
    }
}
fn f() {
    let_ = 'c';
    let_ = 1000;
    let_ = 12E+99_f64;
    let_ = "rust1";
    let_ = -92;
}
"#]],
    );
}

#[test]
fn broken_parenthesis_sequence() {
    check(
        r#"
macro_rules! m1 { ($x:ident) => { ($x } }
macro_rules! m2 { ($x:ident) => {} }

m1!();
m2!(x
"#,
        expect![[r#"
macro_rules! m1 { ($x:ident) => { ($x } }
macro_rules! m2 { ($x:ident) => {} }

/* error: invalid macro definition: expected subtree */
/* error: Failed to lower macro args to token tree */
"#]],
    )
}
