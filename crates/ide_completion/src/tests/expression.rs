//! Completion tests for expressions.
use expect_test::{expect, Expect};

use crate::tests::{completion_list, BASE_ITEMS_FIXTURE};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{}{}", BASE_ITEMS_FIXTURE, ra_fixture));
    expect.assert_eq(&actual)
}

fn check_empty(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual);
}

#[test]
fn completes_various_bindings() {
    check_empty(
        r#"
fn func(param0 @ (param1, param2): (i32, i32)) {
    let letlocal = 92;
    if let ifletlocal = 100 {
        match 0 {
            matcharm => 1 + $0,
            otherwise => (),
        }
    }
    let letlocal2 = 44;
}
"#,
        expect![[r#"
            kw unsafe
            kw match
            kw while
            kw while let
            kw loop
            kw if
            kw if let
            kw for
            kw true
            kw false
            kw return
            kw self
            kw super
            kw crate
            lc matcharm   i32
            lc ifletlocal i32
            lc letlocal   i32
            lc param0     (i32, i32)
            lc param1     i32
            lc param2     i32
            fn func(…)    fn((i32, i32))
            bt u32
        "#]],
    );
}

#[test]
fn completes_all_the_things() {
    cov_mark::check!(unqualified_skip_lifetime_completion);
    check(
        r#"
use non_existant::Unresolved;
mod qualified { pub enum Enum { Variant } }

impl Unit {
    fn foo<'lifetime, TypeParam, const CONST_PARAM: usize>(self) {
        fn local_func() {}
        $0
    }
}
"#,
        expect![[r##"
            kw unsafe
            kw fn
            kw const
            kw type
            kw impl
            kw extern
            kw use
            kw trait
            kw static
            kw mod
            kw match
            kw while
            kw while let
            kw loop
            kw if
            kw if let
            kw for
            kw true
            kw false
            kw let
            kw return
            sn pd
            sn ppd
            kw self
            kw super
            kw crate
            fn local_func() fn()
            bt u32
            lc self         Unit
            tp TypeParam
            cp CONST_PARAM
            sp Self
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            md qualified
            ma makro!(…)    #[macro_export] macro_rules! makro
            ?? Unresolved
            fn function()   fn()
            sc STATIC
            ev TupleV(…)    (u32)
            ct CONST
            ma makro!(…)    #[macro_export] macro_rules! makro
            me self.foo()   fn(self)
        "##]],
    );
}

#[test]
fn shadowing_shows_single_completion() {
    check_empty(
        r#"
fn foo() {
    let bar = 92;
    {
        let bar = 62;
        drop($0)
    }
}
"#,
        // FIXME: should be only one bar here
        expect![[r#"
            kw unsafe
            kw match
            kw while
            kw while let
            kw loop
            kw if
            kw if let
            kw for
            kw true
            kw false
            kw return
            kw self
            kw super
            kw crate
            lc bar       i32
            lc bar       i32
            fn foo()     fn()
            bt u32
        "#]],
    );
}

#[test]
fn in_macro_expr_frag() {
    check_empty(
        r#"
macro_rules! m { ($e:expr) => { $e } }
fn quux(x: i32) {
    m!($0);
}
"#,
        expect![[r#"
            kw unsafe
            kw match
            kw while
            kw while let
            kw loop
            kw if
            kw if let
            kw for
            kw true
            kw false
            kw return
            kw self
            kw super
            kw crate
            bt u32
            lc x         i32
            fn quux(…)   fn(i32)
            ma m!(…)     macro_rules! m
        "#]],
    );
    check_empty(
        r"
macro_rules! m { ($e:expr) => { $e } }
fn quux(x: i32) {
    m!(x$0);
}
",
        expect![[r#"
            kw unsafe
            kw match
            kw while
            kw while let
            kw loop
            kw if
            kw if let
            kw for
            kw true
            kw false
            kw return
            kw self
            kw super
            kw crate
            bt u32
            lc x         i32
            fn quux(…)   fn(i32)
            ma m!(…)     macro_rules! m
        "#]],
    );
    check_empty(
        r#"
macro_rules! m { ($e:expr) => { $e } }
fn quux(x: i32) {
    let y = 92;
    m!(x$0
}
"#,
        expect![[r#"
            kw unsafe
            kw match
            kw while
            kw while let
            kw loop
            kw if
            kw if let
            kw for
            kw true
            kw false
            kw return
            kw self
            kw super
            kw crate
            lc y         i32
            bt u32
            lc x         i32
            fn quux(…)   fn(i32)
            ma m!(…)     macro_rules! m
        "#]],
    );
}
