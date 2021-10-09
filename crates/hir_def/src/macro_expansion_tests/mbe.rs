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
    ($ i:ident) => ( mod $ i {} );
    (= $ i:ident) => ( fn $ i() {} );
    (+ $ i:ident) => ( struct $ i; )
}
m! { foo }
m! { = bar }
m! { + Baz }
"#,
        expect![[r#"
macro_rules! m {
    ($ i:ident) => ( mod $ i {} );
    (= $ i:ident) => ( fn $ i() {} );
    (+ $ i:ident) => ( struct $ i; )
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
    ($ i:ident) => ( mod $ i {} );
    ($ i:ident =) => ( fn $ i() {} );
    ($ i:ident +) => ( struct $ i; )
}
m! { foo }
m! { bar = }
m! { Baz + }
"#,
        expect![[r#"
macro_rules! m {
    ($ i:ident) => ( mod $ i {} );
    ($ i:ident =) => ( fn $ i() {} );
    ($ i:ident +) => ( struct $ i; )
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
    ($ i:ident) => ( mod $ i {} );
    (spam $ i:ident) => ( fn $ i() {} );
    (eggs $ i:ident) => ( struct $ i; )
}
m! { foo }
m! { spam bar }
m! { eggs Baz }
"#,
        expect![[r#"
macro_rules! m {
    ($ i:ident) => ( mod $ i {} );
    (spam $ i:ident) => ( fn $ i() {} );
    (eggs $ i:ident) => ( struct $ i; )
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
    ($ ($ i:ident),*) => ($ ( mod $ i {} )*);
    ($ ($ i:ident)#*) => ($ ( fn $ i() {} )*);
    ($ i:ident ,# $ j:ident) => ( struct $ i; struct $ j; )
}

m! { foo, bar }

m! { foo# bar }

m! { Foo,# Bar }
"#,
        expect![[r##"
macro_rules! m {
    ($ ($ i:ident),*) => ($ ( mod $ i {} )*);
    ($ ($ i:ident)#*) => ($ ( fn $ i() {} )*);
    ($ i:ident ,# $ j:ident) => ( struct $ i; struct $ j; )
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
