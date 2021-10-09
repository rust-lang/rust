//! Tests specific to declarative macros, aka macros by example. This covers
//! both stable `macro_rules!` macros as well as unstable `macro` macros.

mod tt_conversion;
mod matching;
mod meta_syntax;

use expect_test::expect;

use crate::macro_expansion_tests::check;

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
fn match_by_first_token_literally() {
    check(
        r#"
macro_rules! m {
    ($i:ident) => ( mod $i {} );
    (= $i:ident) => ( fn $i() {} );
    (+ $i:ident) => ( struct $i; )
}
m! { foo }
m! { = bar }
m! { + Baz }
"#,
        expect![[r#"
macro_rules! m {
    ($i:ident) => ( mod $i {} );
    (= $i:ident) => ( fn $i() {} );
    (+ $i:ident) => ( struct $i; )
}
mod foo {}
fn bar() {}
struct Baz;
"#]],
    );
}

#[test]
fn match_by_last_token_literally() {
    check(
        r#"
macro_rules! m {
    ($i:ident) => ( mod $i {} );
    ($i:ident =) => ( fn $i() {} );
    ($i:ident +) => ( struct $i; )
}
m! { foo }
m! { bar = }
m! { Baz + }
"#,
        expect![[r#"
macro_rules! m {
    ($i:ident) => ( mod $i {} );
    ($i:ident =) => ( fn $i() {} );
    ($i:ident +) => ( struct $i; )
}
mod foo {}
fn bar() {}
struct Baz;
"#]],
    );
}

#[test]
fn match_by_ident() {
    check(
        r#"
macro_rules! m {
    ($i:ident) => ( mod $i {} );
    (spam $i:ident) => ( fn $i() {} );
    (eggs $i:ident) => ( struct $i; )
}
m! { foo }
m! { spam bar }
m! { eggs Baz }
"#,
        expect![[r#"
macro_rules! m {
    ($i:ident) => ( mod $i {} );
    (spam $i:ident) => ( fn $i() {} );
    (eggs $i:ident) => ( struct $i; )
}
mod foo {}
fn bar() {}
struct Baz;
"#]],
    );
}

#[test]
fn match_by_separator_token() {
    check(
        r#"
macro_rules! m {
    ($($i:ident),*) => ($(mod $i {} )*);
    ($($i:ident)#*) => ($(fn $i() {} )*);
    ($i:ident ,# $ j:ident) => ( struct $i; struct $ j; )
}

m! { foo, bar }

m! { foo# bar }

m! { Foo,# Bar }
"#,
        expect![[r##"
macro_rules! m {
    ($($i:ident),*) => ($(mod $i {} )*);
    ($($i:ident)#*) => ($(fn $i() {} )*);
    ($i:ident ,# $ j:ident) => ( struct $i; struct $ j; )
}

mod foo {}
mod bar {}

fn foo() {}
fn bar() {}

struct Foo;
struct Bar;
"##]],
    );
}

#[test]
fn test_match_group_pattern_with_multiple_defs() {
    check(
        r#"
macro_rules! m {
    ($($i:ident),*) => ( impl Bar { $(fn $i() {})* } );
}
m! { foo, bar }
"#,
        expect![[r#"
macro_rules! m {
    ($($i:ident),*) => ( impl Bar { $(fn $i() {})* } );
}
impl Bar {
    fn foo() {}
    fn bar() {}
}
"#]],
    );
}

#[test]
fn test_match_group_pattern_with_multiple_statement() {
    check(
        r#"
macro_rules! m {
    ($($i:ident),*) => ( fn baz() { $($i ();)* } );
}
m! { foo, bar }
"#,
        expect![[r#"
macro_rules! m {
    ($($i:ident),*) => ( fn baz() { $($i ();)* } );
}
fn baz() {
    foo();
    bar();
}
"#]],
    )
}

#[test]
fn test_match_group_pattern_with_multiple_statement_without_semi() {
    check(
        r#"
macro_rules! m {
    ($($i:ident),*) => ( fn baz() { $($i() );* } );
}
m! { foo, bar }
"#,
        expect![[r#"
macro_rules! m {
    ($($i:ident),*) => ( fn baz() { $($i() );* } );
}
fn baz() {
    foo();
    bar()
}
"#]],
    )
}

#[test]
fn test_match_group_empty_fixed_token() {
    check(
        r#"
macro_rules! m {
    ($($i:ident)* #abc) => ( fn baz() { $($i ();)* } );
}
m!{#abc}
"#,
        expect![[r##"
macro_rules! m {
    ($($i:ident)* #abc) => ( fn baz() { $($i ();)* } );
}
fn baz() {}
"##]],
    )
}

#[test]
fn test_match_group_in_subtree() {
    check(
        r#"
macro_rules! m {
    (fn $name:ident { $($i:ident)* } ) => ( fn $name() { $($i ();)* } );
}
m! { fn baz { a b } }
"#,
        expect![[r#"
macro_rules! m {
    (fn $name:ident { $($i:ident)* } ) => ( fn $name() { $($i ();)* } );
}
fn baz() {
    a();
    b();
}
"#]],
    )
}

#[test]
fn test_expr_order() {
    check(
        r#"
macro_rules! m {
    ($ i:expr) => { fn bar() { $ i * 3; } }
}
// +tree
m! { 1 + 2 }
"#,
        expect![[r#"
macro_rules! m {
    ($ i:expr) => { fn bar() { $ i * 3; } }
}
fn bar() {
    1+2*3;
}
// MACRO_ITEMS@0..15
//   FN@0..15
//     FN_KW@0..2 "fn"
//     NAME@2..5
//       IDENT@2..5 "bar"
//     PARAM_LIST@5..7
//       L_PAREN@5..6 "("
//       R_PAREN@6..7 ")"
//     BLOCK_EXPR@7..15
//       STMT_LIST@7..15
//         L_CURLY@7..8 "{"
//         EXPR_STMT@8..14
//           BIN_EXPR@8..13
//             BIN_EXPR@8..11
//               LITERAL@8..9
//                 INT_NUMBER@8..9 "1"
//               PLUS@9..10 "+"
//               LITERAL@10..11
//                 INT_NUMBER@10..11 "2"
//             STAR@11..12 "*"
//             LITERAL@12..13
//               INT_NUMBER@12..13 "3"
//           SEMICOLON@13..14 ";"
//         R_CURLY@14..15 "}"

"#]],
    )
}
