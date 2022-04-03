//! Completion tests for type position.
use expect_test::{expect, Expect};

use crate::tests::{completion_list, BASE_ITEMS_FIXTURE};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{}\n{}", BASE_ITEMS_FIXTURE, ra_fixture));
    expect.assert_eq(&actual)
}

#[test]
fn record_field_ty() {
    check(
        r#"
struct Foo<'lt, T, const C: usize> {
    f: $0
}
"#,
        expect![[r#"
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
            ma makro!(…) macro_rules! makro
            un Union
            bt u32
        "#]],
    )
}

#[test]
fn tuple_struct_field() {
    check(
        r#"
struct Foo<'lt, T, const C: usize>(f$0);
"#,
        expect![[r#"
            kw pub(crate)
            kw pub(super)
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
            ma makro!(…)  macro_rules! makro
            un Union
            bt u32
        "#]],
    )
}

#[test]
fn fn_return_type() {
    check(
        r#"
fn x<'lt, T, const C: usize>() -> $0
"#,
        expect![[r#"
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
            ma makro!(…) macro_rules! makro
            un Union
            bt u32
        "#]],
    );
}

#[test]
fn inferred_type_const() {
    check(
        r#"
struct Foo<T>(T);
const FOO: $0 = Foo(2);
"#,
        expect![[r#"
            it Foo<i32>
            kw self
            kw super
            kw crate
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Foo<…>
            st Unit
            ma makro!(…) macro_rules! makro
            un Union
            bt u32
        "#]],
    );
}

#[test]
fn inferred_type_closure_param() {
    check(
        r#"
fn f1(f: fn(i32) -> i32) {}
fn f2() {
    f1(|x: $0);
}
"#,
        expect![[r#"
            it i32
            kw self
            kw super
            kw crate
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…) macro_rules! makro
            un Union
            bt u32
        "#]],
    );
}

#[test]
fn inferred_type_closure_return() {
    check(
        r#"
fn f1(f: fn(u64) -> u64) {}
fn f2() {
    f1(|x| -> $0 {
        x + 5
    });
}
"#,
        expect![[r#"
            it u64
            kw self
            kw super
            kw crate
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…) macro_rules! makro
            un Union
            bt u32
        "#]],
    );
}

#[test]
fn inferred_type_fn_return() {
    check(
        r#"
fn f2(x: u64) -> $0 {
    x + 5
}
"#,
        expect![[r#"
            it u64
            kw self
            kw super
            kw crate
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…) macro_rules! makro
            un Union
            bt u32
        "#]],
    );
}

#[test]
fn inferred_type_fn_param() {
    check(
        r#"
fn f1(x: i32) {}
fn f2(x: $0) {
    f1(x);
}
"#,
        expect![[r#"
            it i32
            kw self
            kw super
            kw crate
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…) macro_rules! makro
            un Union
            bt u32
        "#]],
    );
}

#[test]
fn inferred_type_not_in_the_scope() {
    check(
        r#"
mod a {
    pub struct Foo<T>(T);
    pub fn x() -> Foo<Foo<i32>> {
        Foo(Foo(2))
    }
}
fn foo<'lt, T, const C: usize>() {
    let local = ();
    let foo: $0 = a::x();
}
"#,
        expect![[r#"
            it a::Foo<a::Foo<i32>>
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
            ma makro!(…)           macro_rules! makro
            un Union
            md a
            bt u32
        "#]],
    );
}

#[test]
fn inferred_type_let() {
    check(
        r#"
struct Foo<T>(T);
fn foo<'lt, T, const C: usize>() {
    let local = ();
    let foo: $0 = Foo(2);
}
"#,
        expect![[r#"
            it Foo<i32>
            kw self
            kw super
            kw crate
            tp T
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Foo<…>
            st Unit
            ma makro!(…) macro_rules! makro
            un Union
            bt u32
        "#]],
    );
}

#[test]
fn body_type_pos() {
    check(
        r#"
fn foo<'lt, T, const C: usize>() {
    let local = ();
    let _: $0;
}
"#,
        expect![[r#"
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
            ma makro!(…) macro_rules! makro
            un Union
            bt u32
        "#]],
    );
    check(
        r#"
fn foo<'lt, T, const C: usize>() {
    let local = ();
    let _: self::$0;
}
"#,
        expect![[r#"
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…) macro_rules! makro
            un Union
        "#]],
    );
}

#[test]
fn completes_types_and_const_in_arg_list() {
    check(
        r#"
trait Trait2 {
    type Foo;
}

fn foo<'lt, T: Trait2<$0>, const CONST_PARAM: usize>(_: T) {}
"#,
        expect![[r#"
            kw self
            kw super
            kw crate
            ta Foo =  (as Trait2) type Foo
            tp T
            cp CONST_PARAM
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…)          macro_rules! makro
            tt Trait2
            un Union
            ct CONST
            bt u32
        "#]],
    );
    check(
        r#"
trait Trait2 {
    type Foo;
}

fn foo<'lt, T: Trait2<self::$0>, const CONST_PARAM: usize>(_: T) {}
    "#,
        expect![[r#"
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…) macro_rules! makro
            tt Trait2
            un Union
            ct CONST
        "#]],
    );
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
fn func(_: Enum::$0) {}
"#,
        expect![[r#"
            ta AssocType type AssocType = ()
        "#]],
    );
}
