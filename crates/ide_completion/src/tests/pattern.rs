//! Completion tests for pattern position.
use expect_test::{expect, Expect};

use crate::tests::{completion_list, BASE_FIXTURE};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual)
}

fn check_with(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{}\n{}", BASE_FIXTURE, ra_fixture));
    expect.assert_eq(&actual)
}

#[test]
fn ident_rebind_pat() {
    check(
        r#"
fn quux() {
    let en$0 @ x
}
"#,
        expect![[r#"
            kw mut
        "#]],
    );
}

#[test]
fn ident_ref_pat() {
    check(
        r#"
fn quux() {
    let ref en$0
}
"#,
        expect![[r#"
            kw mut
        "#]],
    );
    check(
        r#"
fn quux() {
    let ref en$0 @ x
}
"#,
        expect![[r#"
            kw mut
        "#]],
    );
}

#[test]
fn ident_ref_mut_pat() {
    // FIXME mut is already here, don't complete it again
    check(
        r#"
fn quux() {
    let ref mut en$0
}
"#,
        expect![[r#"
            kw mut
        "#]],
    );
    check(
        r#"
fn quux() {
    let ref mut en$0 @ x
}
"#,
        expect![[r#"
            kw mut
        "#]],
    );
}

#[test]
fn ref_pat() {
    check(
        r#"
fn quux() {
    let &en$0
}
"#,
        expect![[r#"
            kw mut
        "#]],
    );
    // FIXME mut is already here, don't complete it again
    check(
        r#"
fn quux() {
    let &mut en$0
}
"#,
        expect![[r#"
            kw mut
        "#]],
    );
}

#[test]
fn refutable() {
    check_with(
        r#"
fn foo() {
    if let a$0
}
"#,
        expect![[r##"
            kw mut
            en Enum
            bn Record    Record { field$1 }$0
            st Record
            bn Tuple     Tuple($1)$0
            st Tuple
            md module
            st Unit
            ma makro!(…) #[macro_export] macro_rules! makro
            bn TupleV    TupleV($1)$0
            ev TupleV
            ct CONST
            ma makro!(…) #[macro_export] macro_rules! makro
        "##]],
    );
}

#[test]
fn irrefutable() {
    check_with(
        r#"
fn foo() {
   let a$0
}
"#,
        expect![[r##"
            kw mut
            bn Record    Record { field$1 }$0
            st Record
            bn Tuple     Tuple($1)$0
            st Tuple
            st Unit
            ma makro!(…) #[macro_export] macro_rules! makro
            ma makro!(…) #[macro_export] macro_rules! makro
        "##]],
    );
}

#[test]
fn in_param() {
    check_with(
        r#"
fn foo(a$0) {
}
"#,
        expect![[r##"
            kw mut
            bn Record    Record { field$1 }: Record$0
            st Record
            bn Tuple     Tuple($1): Tuple$0
            st Tuple
            st Unit
            ma makro!(…) #[macro_export] macro_rules! makro
            ma makro!(…) #[macro_export] macro_rules! makro
        "##]],
    );
}

#[test]
fn only_fn_like_macros() {
    check(
        r#"
macro_rules! m { ($e:expr) => { $e } }

#[rustc_builtin_macro]
macro Clone {}

fn foo() {
    let x$0
}
"#,
        expect![[r#"
            kw mut
            ma m!(…) macro_rules! m
        "#]],
    );
}

#[test]
fn in_simple_macro_call() {
    check(
        r#"
macro_rules! m { ($e:expr) => { $e } }
enum E { X }

fn foo() {
   m!(match E::X { a$0 })
}
"#,
        expect![[r#"
            kw mut
            ev E::X  ()
            en E
            ma m!(…) macro_rules! m
        "#]],
    );
}

#[test]
fn omits_private_fields_pat() {
    check(
        r#"
mod foo {
    pub struct Record { pub field: i32, _field: i32 }
    pub struct Tuple(pub u32, u32);
    pub struct Invisible(u32, u32);
}
use foo::*;

fn outer() {
    if let a$0
}
"#,
        expect![[r#"
            kw mut
            bn Record    Record { field$1, .. }$0
            st Record
            bn Tuple     Tuple($1, ..)$0
            st Tuple
            st Invisible
            md foo
        "#]],
    )
}

// #[test]
// fn only_shows_ident_completion() {
//     check_edit(
//         "Foo",
//         r#"
// struct Foo(i32);
// fn main() {
//     match Foo(92) {
//         a$0(92) => (),
//     }
// }
// "#,
//         r#"
// struct Foo(i32);
// fn main() {
//     match Foo(92) {
//         Foo(92) => (),
//     }
// }
// "#,
//     );
// }

#[test]
fn completes_self_pats() {
    check(
        r#"
struct Foo(i32);
impl Foo {
    fn foo() {
        match Foo(0) {
            a$0
        }
    }
}
    "#,
        expect![[r#"
            kw mut
            bn Self Self($1)$0
            sp Self
            bn Foo  Foo($1)$0
            st Foo
        "#]],
    )
}

#[test]
fn completes_qualified_variant() {
    check(
        r#"
enum Foo {
    Bar { baz: i32 }
}
impl Foo {
    fn foo() {
        match {Foo::Bar { baz: 0 }} {
            B$0
        }
    }
}
    "#,
        expect![[r#"
            kw mut
            bn Self::Bar Self::Bar { baz$1 }$0
            ev Self::Bar { baz: i32 }
            bn Foo::Bar  Foo::Bar { baz$1 }$0
            ev Foo::Bar  { baz: i32 }
            sp Self
            en Foo
        "#]],
    )
}

#[test]
fn completes_in_record_field_pat() {
    check(
        r#"
struct Foo { bar: Bar }
struct Bar(u32);
fn outer(Foo { bar: $0 }: Foo) {}
"#,
        expect![[r#"
            kw mut
            bn Foo Foo { bar$1 }$0
            st Foo
            bn Bar Bar($1)$0
            st Bar
        "#]],
    )
}

#[test]
fn skips_in_record_field_pat_name() {
    check(
        r#"
struct Foo { bar: Bar }
struct Bar(u32);
fn outer(Foo { bar$0 }: Foo) {}
"#,
        expect![[r#""#]],
    )
}
