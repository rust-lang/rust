//! Completion tests for expressions.
use expect_test::expect;

use crate::tests::check;

#[test]
fn complete_dot_in_attr() {
    check(
        r#"
//- proc_macros: identity
pub struct Foo;
impl Foo {
    fn foo(&self) {}
}

#[proc_macros::identity]
fn main() {
    Foo.$0
}
"#,
        expect![[r#"
            me foo()     fn(&self)
            sn box  Box::new(expr)
            sn call function(expr)
            sn const      const {}
            sn dbg      dbg!(expr)
            sn dbgr    dbg!(&expr)
            sn deref         *expr
            sn let             let
            sn letm        let mut
            sn match match expr {}
            sn ref           &expr
            sn refm      &mut expr
            sn return  return expr
            sn unsafe    unsafe {}
        "#]],
    )
}

#[test]
fn complete_dot_in_attr2() {
    check(
        r#"
//- proc_macros: identity
pub struct Foo;
impl Foo {
    fn foo(&self) {}
}

#[proc_macros::identity]
fn main() {
    Foo.f$0
}
"#,
        expect![[r#"
            me foo()     fn(&self)
            sn box  Box::new(expr)
            sn call function(expr)
            sn const      const {}
            sn dbg      dbg!(expr)
            sn dbgr    dbg!(&expr)
            sn deref         *expr
            sn let             let
            sn letm        let mut
            sn match match expr {}
            sn ref           &expr
            sn refm      &mut expr
            sn return  return expr
            sn unsafe    unsafe {}
        "#]],
    )
}

#[test]
fn complete_dot_in_attr_input() {
    check(
        r#"
//- proc_macros: input_replace
pub struct Foo;
impl Foo {
    fn foo(&self) {}
}

#[proc_macros::input_replace(
    fn surprise() {
        Foo.$0
    }
)]
fn main() {}
"#,
        expect![[r#"
            me foo()     fn(&self)
            sn box  Box::new(expr)
            sn call function(expr)
            sn const      const {}
            sn dbg      dbg!(expr)
            sn dbgr    dbg!(&expr)
            sn deref         *expr
            sn let             let
            sn letm        let mut
            sn match match expr {}
            sn ref           &expr
            sn refm      &mut expr
            sn return  return expr
            sn unsafe    unsafe {}
        "#]],
    )
}

#[test]
fn complete_dot_in_attr_input2() {
    check(
        r#"
//- proc_macros: input_replace
pub struct Foo;
impl Foo {
    fn foo(&self) {}
}

#[proc_macros::input_replace(
    fn surprise() {
        Foo.f$0
    }
)]
fn main() {}
"#,
        expect![[r#"
            me foo()     fn(&self)
            sn box  Box::new(expr)
            sn call function(expr)
            sn const      const {}
            sn dbg      dbg!(expr)
            sn dbgr    dbg!(&expr)
            sn deref         *expr
            sn let             let
            sn letm        let mut
            sn match match expr {}
            sn ref           &expr
            sn refm      &mut expr
            sn return  return expr
            sn unsafe    unsafe {}
        "#]],
    )
}

#[test]
fn issue_13836_str() {
    check(
        r#"
//- proc_macros: shorten
fn main() {
    let s = proc_macros::shorten!("text.$0");
}
"#,
        expect![[r#""#]],
    )
}

#[test]
fn issue_13836_ident() {
    check(
        r#"
//- proc_macros: shorten
struct S;
impl S {
    fn foo(&self) {}
}
fn main() {
    let s = proc_macros::shorten!(S.fo$0);
}
"#,
        expect![[r#""#]],
    )
}
