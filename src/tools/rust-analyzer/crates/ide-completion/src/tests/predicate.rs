//! Completion tests for predicates and bounds.
use expect_test::{expect, Expect};

use crate::tests::{check_empty, completion_list, BASE_ITEMS_FIXTURE};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{BASE_ITEMS_FIXTURE}\n{ra_fixture}"));
    expect.assert_eq(&actual)
}

#[test]
fn predicate_start() {
    // FIXME: `for` kw
    check(
        r#"
struct Foo<'lt, T, const C: usize> where $0 {}
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
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn bound_for_type_pred() {
    check(
        r#"
struct Foo<'lt, T, const C: usize> where T: $0 {}
"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            tt Trait
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn bound_for_lifetime_pred() {
    // FIXME: should only show lifetimes here, that is we shouldn't get any completions here when not typing
    // a `'`
    check(
        r#"
struct Foo<'lt, T, const C: usize> where 'lt: $0 {}
"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            tt Trait
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn bound_for_for_pred() {
    check(
        r#"
struct Foo<'lt, T, const C: usize> where for<'a> T: $0 {}
"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            tt Trait
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn param_list_for_for_pred() {
    check(
        r#"
struct Foo<'lt, T, const C: usize> where for<'a> $0 {}
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
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn pred_on_fn_in_impl() {
    check(
        r#"
impl Record {
    fn method(self) where $0 {}
}
"#,
        expect![[r#"
            en Enum
            ma makro!(…) macro_rules! makro
            md module
            sp Self
            st Record
            st Tuple
            st Unit
            tt Trait
            un Union
            bt u32
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn pred_no_unstable_item_on_stable() {
    check_empty(
        r#"
//- /main.rs crate:main deps:std
use std::*;
struct Foo<T> where T: $0 {}
//- /std.rs crate:std
#[unstable]
pub trait Trait {}
"#,
        expect![[r#"
            md std
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn pred_unstable_item_on_nightly() {
    check_empty(
        r#"
//- toolchain:nightly
//- /main.rs crate:main deps:std
use std::*;
struct Foo<T> where T: $0 {}
//- /std.rs crate:std
#[unstable]
pub trait Trait {}
"#,
        expect![[r#"
            md std
            tt Trait
            kw crate::
            kw self::
        "#]],
    );
}
