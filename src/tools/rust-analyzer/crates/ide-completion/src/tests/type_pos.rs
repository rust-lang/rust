//! Completion tests for type position.
use expect_test::{expect, Expect};

use crate::tests::{check_empty, completion_list, BASE_ITEMS_FIXTURE};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{BASE_ITEMS_FIXTURE}\n{ra_fixture}"));
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
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            sp Self
            st Foo<…>
            st Record
            st Tuple
            st Unit
            tt Trait
            tp T
            un Union
            bt u32
            kw crate::
            kw self::
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
            en Enum
            ma makro!(…)  macro_rules! makro
            md module
            sp Self
            st Foo<…>
            st Record
            st Tuple
            st Unit
            tt Trait
            tp T
            un Union
            bt u32
            kw crate::
            kw pub
            kw pub(crate)
            kw pub(super)
            kw self::
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
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt Trait
            tp T
            un Union
            bt u32
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn fn_return_type_no_local_items() {
    check(
        r#"
fn foo() -> B$0 {
    struct Bar;
    enum Baz {}
    union Bax {
        i: i32,
        f: f32
    }
}
"#,
        expect![[r#"
        en Enum
        ma makro!(…) macro_rules! makro
        md module
        st Record
        st Tuple
        st Unit
        tt Trait
        un Union
        bt u32
        it ()
        kw crate::
        kw self::
    "#]],
    )
}

#[test]
fn inferred_type_const() {
    check(
        r#"
struct Foo<T>(T);
const FOO: $0 = Foo(2);
"#,
        expect![[r#"
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Foo<…>
            st Record
            st Tuple
            st Unit
            tt Trait
            un Union
            bt u32
            it Foo<i32>
            kw crate::
            kw self::
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
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt Trait
            un Union
            bt u32
            it i32
            kw crate::
            kw self::
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
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt Trait
            un Union
            bt u32
            it u64
            kw crate::
            kw self::
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
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt Trait
            un Union
            bt u32
            it u64
            kw crate::
            kw self::
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
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt Trait
            un Union
            bt u32
            it i32
            kw crate::
            kw self::
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
            en Enum
            ma makro!(…)           macro_rules! makro
            md a
            md module
            st Record
            st Tuple
            st Unit
            tt Trait
            tp T
            un Union
            bt u32
            it a::Foo<a::Foo<i32>>
            kw crate::
            kw self::
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
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Foo<…>
            st Record
            st Tuple
            st Unit
            tt Trait
            tp T
            un Union
            bt u32
            it Foo<i32>
            kw crate::
            kw self::
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
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt Trait
            tp T
            un Union
            bt u32
            kw crate::
            kw self::
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
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt Trait
            un Union
        "#]],
    );
}

#[test]
fn completes_types_and_const_in_arg_list() {
    cov_mark::check!(complete_assoc_type_in_generics_list);
    check(
        r#"
trait Trait1 {
    type Super;
}
trait Trait2: Trait1 {
    type Foo;
}

fn foo<'lt, T: Trait2<$0>, const CONST_PARAM: usize>(_: T) {}
"#,
        expect![[r#"
            ta Foo =  (as Trait2)   type Foo
            ta Super =  (as Trait1) type Super
        "#]],
    );
    check(
        r#"
trait Trait1 {
    type Super;
}
trait Trait2<T>: Trait1 {
    type Foo;
}

fn foo<'lt, T: Trait2<$0>, const CONST_PARAM: usize>(_: T) {}
"#,
        expect![[r#"
            ct CONST
            cp CONST_PARAM
            en Enum
            ma makro!(…)   macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt Trait
            tt Trait1
            tt Trait2
            tp T
            un Union
            bt u32
            kw crate::
            kw self::
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
            ct CONST
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt Trait
            tt Trait2
            un Union
        "#]],
    );
}

#[test]
fn no_assoc_completion_outside_type_bounds() {
    check(
        r#"
struct S;
trait Tr<T> {
    type Ty;
}

impl Tr<$0
    "#,
        expect![[r#"
            ct CONST
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            sp Self
            st Record
            st S
            st Tuple
            st Unit
            tt Tr
            tt Trait
            un Union
            bt u32
            kw crate::
            kw self::
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

#[test]
fn completes_type_parameter_or_associated_type() {
    check(
        r#"
trait MyTrait<T, U> {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<u$0
"#,
        expect![[r#"
            ct CONST
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt MyTrait
            tt Trait
            un Union
            bt u32
            kw crate::
            kw self::
        "#]],
    );

    check(
        r#"
trait MyTrait<T, U> {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<u8, u$0
"#,
        expect![[r#"
            ct CONST
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt MyTrait
            tt Trait
            un Union
            bt u32
            kw crate::
            kw self::
        "#]],
    );

    check(
        r#"
trait MyTrait<T, U> {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<u8, u8, I$0
"#,
        expect![[r#"
            ta Item1 =  (as MyTrait) type Item1
            ta Item2 =  (as MyTrait) type Item2
        "#]],
    );
}

#[test]
fn completes_type_parameter_or_associated_type_with_default_value() {
    check(
        r#"
trait MyTrait<T, U = u8> {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<u$0
"#,
        expect![[r#"
            ct CONST
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt MyTrait
            tt Trait
            un Union
            bt u32
            kw crate::
            kw self::
        "#]],
    );

    check(
        r#"
trait MyTrait<T, U = u8> {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<u8, u$0
"#,
        expect![[r#"
            ct CONST
            en Enum
            ma makro!(…)             macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt MyTrait
            tt Trait
            ta Item1 =  (as MyTrait) type Item1
            ta Item2 =  (as MyTrait) type Item2
            un Union
            bt u32
            kw crate::
            kw self::
        "#]],
    );

    check(
        r#"
trait MyTrait<T, U = u8> {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<u8, u8, I$0
"#,
        expect![[r#"
            ta Item1 =  (as MyTrait) type Item1
            ta Item2 =  (as MyTrait) type Item2
        "#]],
    );
}

#[test]
fn completes_types_after_associated_type() {
    check(
        r#"
trait MyTrait {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<Item1 = $0
"#,
        expect![[r#"
            ct CONST
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt MyTrait
            tt Trait
            un Union
            bt u32
            kw crate::
            kw self::
        "#]],
    );

    check(
        r#"
trait MyTrait {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<Item1 = u8, Item2 = $0
"#,
        expect![[r#"
            ct CONST
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            tt MyTrait
            tt Trait
            un Union
            bt u32
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn type_pos_no_unstable_type_on_stable() {
    check_empty(
        r#"
//- /main.rs crate:main deps:std
use std::*;
struct Foo {
    f: $0
}
//- /std.rs crate:std
#[unstable]
pub struct S;
"#,
        expect![[r#"
            md std
            sp Self
            st Foo
            bt u32
            kw crate::
            kw self::
        "#]],
    )
}

#[test]
fn type_pos_unstable_type_on_nightly() {
    check_empty(
        r#"
//- toolchain:nightly
//- /main.rs crate:main deps:std
use std::*;
struct Foo {
    f: $0
}
//- /std.rs crate:std
#[unstable]
pub struct S;
"#,
        expect![[r#"
            md std
            sp Self
            st Foo
            st S
            bt u32
            kw crate::
            kw self::
        "#]],
    )
}
