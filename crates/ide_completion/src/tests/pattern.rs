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
    check_empty(
        r#"
fn foo() {
    for &$0 in () {}
}
"#,
        expect![[r#"
            kw mut
        "#]],
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
            md module
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
            md module
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
            md module
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
            ct ASSOC_CONST const ASSOC_CONST: ()
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

#[test]
fn completes_no_delims_if_existing() {
    check_empty(
        r#"
struct Bar(u32);
fn foo() {
    match Bar(0) {
        B$0(b) => {}
    }
}
"#,
        expect![[r#"
            kw self::
            kw super::
            kw crate::
        "#]],
    );
    check_empty(
        r#"
struct Foo { bar: u32 }
fn foo() {
    match Foo { bar: 0 } {
        F$0 { bar } => {}
    }
}
"#,
        expect![[r#"
            kw return
            kw self
            kw super
            kw crate
            st Foo
            fn foo()  fn()
            bt u32
        "#]],
    );
    check_empty(
        r#"
enum Enum {
    TupleVariant(u32)
}
fn foo() {
    match Enum::TupleVariant(0) {
        Enum::T$0(b) => {}
    }
}
"#,
        expect![[r#"
            ev TupleVariant(…) TupleVariant
        "#]],
    );
    check_empty(
        r#"
enum Enum {
    RecordVariant { field: u32 }
}
fn foo() {
    match (Enum::RecordVariant { field: 0 }) {
        Enum::RecordV$0 { field } => {}
    }
}
"#,
        expect![[r#"
            ev RecordVariant {…} RecordVariant
        "#]],
    );
}

#[test]
fn completes_associated_const() {
    check_empty(
        r#"
#[derive(PartialEq, Eq)]
struct Ty(u8);

impl Ty {
    const ABC: Self = Self(0);
}

fn f(t: Ty) {
    match t {
        Ty::$0 => {}
        _ => {}
    }
}
"#,
        expect![[r#"
            ct ABC const ABC: Self
        "#]],
    );

    check_empty(
        r#"
enum MyEnum {}

impl MyEnum {
    pub const A: i32 = 123;
    pub const B: i32 = 456;
}

fn f(e: MyEnum) {
    match e {
        MyEnum::$0 => {}
        _ => {}
    }
}
"#,
        expect![[r#"
            ct A pub const A: i32
            ct B pub const B: i32
        "#]],
    );

    check_empty(
        r#"
union U {
    i: i32,
    f: f32,
}

impl U {
    pub const C: i32 = 123;
    pub const D: i32 = 456;
}

fn f(u: U) {
    match u {
        U::$0 => {}
        _ => {}
    }
}
"#,
        expect![[r#"
            ct C pub const C: i32
            ct D pub const D: i32
        "#]],
    );

    check_empty(
        r#"
#[lang = "u32"]
impl u32 {
    pub const MIN: Self = 0;
}

fn f(v: u32) {
    match v {
        u32::$0
    }
}
        "#,
        expect![[r#"
            ct MIN pub const MIN: Self
        "#]],
    );
}
