//! Completion tests for pattern position.
use expect_test::{expect, Expect};

use crate::tests::{check_edit, completion_list, BASE_ITEMS_FIXTURE};

fn check_empty(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual)
}

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{}\n{}", BASE_ITEMS_FIXTURE, ra_fixture));
    expect.assert_eq(&actual)
}

#[test]
fn wildcard() {
    check(
        r#"
fn quux() {
    let _$0
}
"#,
        expect![""],
    );
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
            kw mut
            kw ref
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
            ct CONST
            en Enum
            ma makro!(…)  macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            ev TupleV
            bn Record {…} Record { field$1 }$0
            bn Tuple(…)   Tuple($1)$0
            bn TupleV(…)  TupleV($1)$0
            kw mut
            kw ref
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
            en SingleVariantEnum
            ma makro!(…)         macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            ev Variant
            bn Record {…}        Record { field$1 }$0
            bn Tuple(…)          Tuple($1)$0
            bn Variant           Variant$0
            kw mut
            kw ref
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
            ma makro!(…)  macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            bn Record {…} Record { field$1 }: Record$0
            bn Tuple(…)   Tuple($1): Tuple$0
            kw mut
            kw ref
        "#]],
    );
    check(
        r#"
fn foo(a$0: Tuple) {
}
"#,
        expect![[r#"
            ma makro!(…)  macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            bn Record {…} Record { field$1 }$0
            bn Tuple(…)   Tuple($1)$0
            kw mut
            kw ref
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
            ma m!(…) macro_rules! m
            kw mut
            kw ref
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
            en E
            ma m!(…) macro_rules! m
            bn E::X  E::X$0
            kw mut
            kw ref
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
            md foo
            st Invisible
            st Record
            st Tuple
            bn Record {…} Record { field$1, .. }$0
            bn Tuple(…)   Tuple($1, ..)$0
            kw mut
            kw ref
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
            sp Self
            st Foo
            bn Foo(…)  Foo($1)$0
            bn Self(…) Self($1)$0
            kw mut
            kw ref
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
            ct ASSOC_CONST const ASSOC_CONST: ()
            bn RecordV {…} RecordV { field$1 }$0
            bn TupleV(…)   TupleV($1)$0
            bn UnitV       UnitV$0
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
            st Bar
            st Foo
            bn Bar(…)  Bar($1)$0
            bn Foo {…} Foo { bar$1 }$0
            kw mut
            kw ref
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
        expect![[r#"
            kw mut
            kw ref
        "#]],
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
            st Bar
            st Foo
            bn Bar(…)  Bar($1): Bar$0
            bn Foo {…} Foo { bar$1 }: Foo$0
            kw mut
            kw ref
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
            st Bar
            st Foo
            bn Bar(…)  Bar($1)$0
            bn Foo {…} Foo { bar$1 }$0
            kw mut
            kw ref
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
            st Bar
            kw crate::
            kw self::
        "#]],
    );
    check_empty(
        r#"
struct Foo { bar: u32 }
fn foo() {
    match (Foo { bar: 0 }) {
        F$0 { bar } => {}
    }
}
"#,
        expect![[r#"
            st Foo
            kw crate::
            kw self::
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
            bn TupleVariant TupleVariant
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
            bn RecordVariant RecordVariant
        "#]],
    );
}

#[test]
fn completes_enum_variant_pat() {
    cov_mark::check!(enum_variant_pattern_path);
    check_edit(
        "RecordVariant{}",
        r#"
enum Enum {
    RecordVariant { field: u32 }
}
fn foo() {
    match (Enum::RecordVariant { field: 0 }) {
        Enum::RecordV$0
    }
}
"#,
        r#"
enum Enum {
    RecordVariant { field: u32 }
}
fn foo() {
    match (Enum::RecordVariant { field: 0 }) {
        Enum::RecordVariant { field$1 }$0
    }
}
"#,
    );
}

#[test]
fn completes_enum_variant_pat_escape() {
    cov_mark::check!(enum_variant_pattern_path);
    check_empty(
        r#"
enum Enum {
    A,
    B { r#type: i32 },
    r#type,
    r#struct { r#type: i32 },
}
fn foo() {
    match (Enum::A) {
        $0
    }
}
"#,
        expect![[r#"
            en Enum
            bn Enum::A          Enum::A$0
            bn Enum::B {…}      Enum::B { r#type$1 }$0
            bn Enum::struct {…} Enum::r#struct { r#type$1 }$0
            bn Enum::type       Enum::r#type$0
            kw mut
            kw ref
        "#]],
    );

    check_empty(
        r#"
enum Enum {
    A,
    B { r#type: i32 },
    r#type,
    r#struct { r#type: i32 },
}
fn foo() {
    match (Enum::A) {
        Enum::$0
    }
}
"#,
        expect![[r#"
            bn A          A$0
            bn B {…}      B { r#type$1 }$0
            bn struct {…} r#struct { r#type$1 }$0
            bn type       r#type$0
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

#[test]
fn in_method_param() {
    check_empty(
        r#"
struct Ty(u8);

impl Ty {
    fn foo($0)
}
"#,
        expect![[r#"
            sp Self
            st Ty
            bn &mut self
            bn &self
            bn Self(…)   Self($1): Self$0
            bn Ty(…)     Ty($1): Ty$0
            bn mut self
            bn self
            kw mut
            kw ref
        "#]],
    );
    check_empty(
        r#"
struct Ty(u8);

impl Ty {
    fn foo(s$0)
}
"#,
        expect![[r#"
            sp Self
            st Ty
            bn &mut self
            bn &self
            bn Self(…)   Self($1): Self$0
            bn Ty(…)     Ty($1): Ty$0
            bn mut self
            bn self
            kw mut
            kw ref
        "#]],
    );
    check_empty(
        r#"
struct Ty(u8);

impl Ty {
    fn foo(s$0, foo: u8)
}
"#,
        expect![[r#"
            sp Self
            st Ty
            bn &mut self
            bn &self
            bn Self(…)   Self($1): Self$0
            bn Ty(…)     Ty($1): Ty$0
            bn mut self
            bn self
            kw mut
            kw ref
        "#]],
    );
    check_empty(
        r#"
struct Ty(u8);

impl Ty {
    fn foo(foo: u8, b$0)
}
"#,
        expect![[r#"
            sp Self
            st Ty
            bn Self(…) Self($1): Self$0
            bn Ty(…)   Ty($1): Ty$0
            kw mut
            kw ref
        "#]],
    );
}
