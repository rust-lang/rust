//! Unlike rustc, rust-analyzer's syntax tree are not "made of" token trees.
//! Rather, token trees are an explicit bridge between the parser and
//! (procedural or declarative) macros.
//!
//! This module tests tt <-> syntax tree conversion specifically. In particular,
//! it, among other things, check that we convert `tt` to the right kind of
//! syntax node depending on the macro call-site.
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
    let _ = 'c';
    let _ = 1000;
    let _ = 12E+99_f64;
    let _ = "rust1";
    let _ = -92;
}
"#]],
    );
}

#[test]
fn roundtrip_lifetime() {
    check(
        r#"
macro_rules! m {
    ($($t:tt)*) => { $($t)*}
}
m!(static bar: &'static str = "hello";);
"#,
        expect![[r#"
macro_rules! m {
    ($($t:tt)*) => { $($t)*}
}
static bar: &'static str = "hello";
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
/* error: invalid token tree */
"#]],
    )
}

#[test]
fn expansion_does_not_parse_as_expression() {
    check(
        r#"
macro_rules! stmts {
    () => { fn foo() {} }
}

fn f() { let _ = stmts!/*+errors*/(); }
"#,
        expect![[r#"
macro_rules! stmts {
    () => { fn foo() {} }
}

fn f() { let _ = /* parse error: expected expression */
fn foo() {}; }
"#]],
    )
}

#[test]
fn broken_pat() {
    check(
        r#"
macro_rules! m1 { () => (Some(x) left overs) }
macro_rules! m2 { () => ($) }

fn main() {
    let m1!() = ();
    let m2!/*+errors*/() = ();
}
"#,
        expect![[r#"
macro_rules! m1 { () => (Some(x) left overs) }
macro_rules! m2 { () => ($) }

fn main() {
    let Some(x)left overs = ();
    let /* parse error: expected pattern */
$ = ();
}
"#]],
    )
}

#[test]
fn float_literal_in_tt() {
    check(
        r#"
macro_rules! constant {
    ($( $ret:expr; )*) => {};
}
macro_rules! float_const_impl {
    () => ( constant!(0.3; 3.3;); );
}
float_const_impl! {}
"#,
        expect![[r#"
macro_rules! constant {
    ($( $ret:expr; )*) => {};
}
macro_rules! float_const_impl {
    () => ( constant!(0.3; 3.3;); );
}
constant!(0.3;
3.3;
);
"#]],
    );
}

#[test]
fn float_literal_in_output() {
    check(
        r#"
macro_rules! constant {
    ($e:expr ;) => {$e};
}

const _: () = constant!(0.0;);
const _: () = constant!(0.;);
const _: () = constant!(0e0;);
"#,
        expect![[r#"
macro_rules! constant {
    ($e:expr ;) => {$e};
}

const _: () = 0.0;
const _: () = 0.;
const _: () = 0e0;
"#]],
    );
}
