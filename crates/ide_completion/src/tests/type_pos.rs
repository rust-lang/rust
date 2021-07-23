//! Completion tests for type position.
use expect_test::{expect, Expect};

use crate::tests::{completion_list, BASE_ITEMS_FIXTURE};

fn check_with(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{}\n{}", BASE_ITEMS_FIXTURE, ra_fixture));
    expect.assert_eq(&actual)
}

#[test]
fn record_field_ty() {
    check_with(
        r#"
struct Foo<'lt, T, const C: usize> {
    f: $0
}
"#,
        expect![[r##"
            kw self
            kw super
            kw crate
            sp Self
            tp T
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Foo<…>
            st Unit
            ma makro!(…) #[macro_export] macro_rules! makro
            ma makro!(…) #[macro_export] macro_rules! makro
            bt u32
        "##]],
    )
}

#[test]
fn tuple_struct_field() {
    check_with(
        r#"
struct Foo<'lt, T, const C: usize>(f$0);
"#,
        expect![[r##"
            kw pub(crate)
            kw pub
            kw self
            kw super
            kw crate
            sp Self
            tp T
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Foo<…>
            st Unit
            ma makro!(…)  #[macro_export] macro_rules! makro
            ma makro!(…)  #[macro_export] macro_rules! makro
            bt u32
        "##]],
    )
}

#[test]
fn fn_return_type() {
    check_with(
        r#"
fn x<'lt, T, const C: usize>() -> $0
"#,
        expect![[r##"
            kw self
            kw super
            kw crate
            tp T
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…) #[macro_export] macro_rules! makro
            ma makro!(…) #[macro_export] macro_rules! makro
            bt u32
        "##]],
    );
}

#[test]
fn body_type_pos() {
    check_with(
        r#"
fn foo<'lt, T, const C: usize>() {
    let local = ();
    let _: $0;
}
"#,
        expect![[r##"
            kw self
            kw super
            kw crate
            tp T
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…) #[macro_export] macro_rules! makro
            ma makro!(…) #[macro_export] macro_rules! makro
            bt u32
        "##]],
    );
    check_with(
        r#"
fn foo<'lt, T, const C: usize>() {
    let local = ();
    let _: self::$0;
}
"#,
        expect![[r##"
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…) #[macro_export] macro_rules! makro
        "##]],
    );
}

#[test]
fn completes_types_and_const_in_arg_list() {
    check_with(
        r#"
trait Trait2 {
    type Foo;
}

fn foo<'lt, T: Trait2<$0>, const CONST_PARAM: usize>(_: T) {}
"#,
        expect![[r##"
            kw self
            kw super
            kw crate
            ta Foo =  (as Trait2) type Foo;
            tp T
            cp CONST_PARAM
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…)          #[macro_export] macro_rules! makro
            tt Trait2
            ct CONST
            ma makro!(…)          #[macro_export] macro_rules! makro
            bt u32
        "##]],
    );
    check_with(
        r#"
trait Trait2 {
    type Foo;
}

fn foo<'lt, T: Trait2<self::$0>, const CONST_PARAM: usize>(_: T) {}
    "#,
        expect![[r##"
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…) #[macro_export] macro_rules! makro
            tt Trait2
            ct CONST
        "##]],
    );
}
