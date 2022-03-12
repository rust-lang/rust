//! Completion tests for pattern position.
use expect_test::{expect, Expect};

use crate::tests::{completion_list, BASE_ITEMS_FIXTURE};

fn check_empty(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual)
}

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{}\n{}", BASE_ITEMS_FIXTURE, ra_fixture));
    expect.assert_eq(&actual)
}

#[test]
fn ident_rebind_pat() {
    check_empty(
        r#"
fn quux() {
    let en$0 @ x
}
"#,
        expect![[r#"
            kw ref
            kw mut
        "#]],
    );
}

#[test]
fn ident_ref_pat() {
    check_empty(
        r#"
fn quux() {
    let ref en$0
}
"#,
        expect![[r#"
            kw mut
        "#]],
    );
    check_empty(
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
    check_empty(
        r#"
fn quux() {
    let ref mut en$0
}
"#,
        expect![[r#""#]],
    );
    check_empty(
        r#"
fn quux() {
    let ref mut en$0 @ x
}
"#,
        expect![[r#""#]],
    );
}

#[test]
fn ref_pat() {
    check_empty(
        r#"
fn quux() {
    let &en$0
}
"#,
        expect![[r#"
            kw mut
        "#]],
    );
    check_empty(
        r#"
fn quux() {
    let &mut en$0
}
"#,
        expect![[r#""#]],
    );
}

#[test]
fn refutable() {
    check(
        r#"
fn foo() {
    if let a$0
}
"#,
        expect![[r#"
            kw ref
            kw mut
            en Enum
            bn Record    Record { field$1 }$0
            st Record
            bn Tuple     Tuple($1)$0
            st Tuple
            md module
            st Unit
            ma makro!(…) macro_rules! makro
            bn TupleV    TupleV($1)$0
            ev TupleV
            ct CONST
        "#]],
    );
}

#[test]
fn irrefutable() {
    check(
        r#"
enum SingleVariantEnum {
    Variant
}
use SingleVariantEnum::Variant;
fn foo() {
   let a$0
}
"#,
        expect![[r#"
            kw ref
            kw mut
            bn Record            Record { field$1 }$0
            st Record
            bn Tuple             Tuple($1)$0
            st Tuple
            ev Variant
            en SingleVariantEnum
            st Unit
            ma makro!(…)         macro_rules! makro
        "#]],
    );
}

#[test]
fn in_param() {
    check(
        r#"
fn foo(a$0) {
}
"#,
        expect![[r#"
            kw ref
            kw mut
            bn Record    Record { field$1 }: Record$0
            st Record
            bn Tuple     Tuple($1): Tuple$0
            st Tuple
            st Unit
            ma makro!(…) macro_rules! makro
        "#]],
    );
    check(
        r#"
fn foo(a$0: Tuple) {
}
"#,
        expect![[r#"
            kw ref
            kw mut
            bn Record    Record { field$1 }$0
            st Record
            bn Tuple     Tuple($1)$0
            st Tuple
            st Unit
            ma makro!(…) macro_rules! makro
        "#]],
    );
}

#[test]
fn only_fn_like_macros() {
    check_empty(
        r#"
macro_rules! m { ($e:expr) => { $e } }

#[rustc_builtin_macro]
macro Clone {}

fn foo() {
    let x$0
}
"#,
        expect![[r#"
            kw ref
            kw mut
            ma m!(…) macro_rules! m
        "#]],
    );
}

#[test]
fn in_simple_macro_call() {
    check_empty(
        r#"
macro_rules! m { ($e:expr) => { $e } }
enum E { X }

fn foo() {
   m!(match E::X { a$0 })
}
"#,
        expect![[r#"
            kw ref
            kw mut
            ev E::X  E::X
            en E
            ma m!(…) macro_rules! m
        "#]],
    );
}

#[test]
fn omits_private_fields_pat() {
    check_empty(
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
            kw ref
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

#[test]
fn completes_self_pats() {
    check_empty(
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
            kw ref
            kw mut
            bn Self Self($1)$0
            sp Self
            bn Foo  Foo($1)$0
            st Foo
        "#]],
    )
}

#[test]
fn enum_qualified() {
    check(
        r#"
impl Enum {
    type AssocType = ();
    const ASSOC_CONST: () = ();
    fn assoc_fn() {}
}
fn func() {
    if let Enum::$0 = unknown {}
}
"#,
        expect![[r#"
            ev TupleV(…)   TupleV(u32)
            ev RecordV {…} RecordV { field: u32 }
            ev UnitV       UnitV
        "#]],
    );
}

#[test]
fn completes_in_record_field_pat() {
    check_empty(
        r#"
struct Foo { bar: Bar }
struct Bar(u32);
fn outer(Foo { bar: $0 }: Foo) {}
"#,
        expect![[r#"
            kw ref
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
    check_empty(
        r#"
struct Foo { bar: Bar }
struct Bar(u32);
fn outer(Foo { bar$0 }: Foo) {}
"#,
        expect![[r#""#]],
    )
}

#[test]
fn completes_in_fn_param() {
    check_empty(
        r#"
struct Foo { bar: Bar }
struct Bar(u32);
fn foo($0) {}
"#,
        expect![[r#"
            kw ref
            kw mut
            bn Foo Foo { bar$1 }: Foo$0
            st Foo
            bn Bar Bar($1): Bar$0
            st Bar
        "#]],
    )
}

#[test]
fn completes_in_closure_param() {
    check_empty(
        r#"
struct Foo { bar: Bar }
struct Bar(u32);
fn foo() {
    |$0| {};
}
"#,
        expect![[r#"
            kw ref
            kw mut
            bn Foo Foo { bar$1 }$0
            st Foo
            bn Bar Bar($1)$0
            st Bar
        "#]],
    )
}
