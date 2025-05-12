//! Completion tests for predicates and bounds.
use expect_test::expect;

use crate::tests::{check, check_with_base_items};

#[test]
fn predicate_start() {
    // FIXME: `for` kw
    check_with_base_items(
        r#"
struct Foo<'lt, T, const C: usize> where $0 {}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Foo<…> Foo<'_, {unknown}, _>
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn bound_for_type_pred() {
    check_with_base_items(
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
    check_with_base_items(
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
    check_with_base_items(
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
    check_with_base_items(
        r#"
struct Foo<'lt, T, const C: usize> where for<'a> $0 {}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Foo<…> Foo<'_, {unknown}, _>
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn pred_on_fn_in_impl() {
    check_with_base_items(
        r#"
impl Record {
    fn method(self) where $0 {}
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            sp Self                  Record
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn pred_no_unstable_item_on_stable() {
    check(
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
    check(
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
