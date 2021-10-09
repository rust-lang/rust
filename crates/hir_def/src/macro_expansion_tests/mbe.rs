use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn wrong_nesting_level() {
    check(
        r#"
macro_rules! m {
    ($($i:ident);*) => ($i)
}
m!{a}
"#,
        expect![[r#"
macro_rules! m {
    ($($i:ident);*) => ($i)
}
/* error: expected simple binding, found nested binding `i` */
"#]],
    );
}

#[test]
fn expansion_does_not_parse_as_expression() {
    check(
        r#"
macro_rules! stmts {
    () => { let _ = 0; }
}

fn f() { let _ = stmts!(); }
"#,
        expect![[r#"
macro_rules! stmts {
    () => { let _ = 0; }
}

fn f() { let _ = /* error: could not convert tokens */; }
"#]],
    )
}

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

/* error: Failed to find macro definition */
/* error: Failed to lower macro args to token tree */
"#]],
    )
}

#[test]
fn unary_minus_is_a_literal() {
    check(
        r#"
macro_rules! m { ($x:literal) => (literal!()); ($x:tt) => (not_a_literal!()); }
m!(92);
m!(-92);
m!(-9.2);
m!(--92);
"#,
        expect![[r#"
macro_rules! m { ($x:literal) => (literal!()); ($x:tt) => (not_a_literal!()); }
literal!()
literal!()
literal!()
/* error: leftover tokens */not_a_literal!()
"#]],
    )
}
