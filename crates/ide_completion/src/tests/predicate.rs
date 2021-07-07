//! Completion tests for predicates and bounds.
use expect_test::{expect, Expect};

use crate::tests::{completion_list, BASE_FIXTURE};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{}\n{}", BASE_FIXTURE, ra_fixture));
    expect.assert_eq(&actual)
}

#[test]
fn predicate_start() {
    // FIXME: `for` kw
    check(
        r#"
struct Foo<'lt, T, const C: usize> where $0 {}
"#,
        expect![[r##"
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
            ma makro!(…) #[macro_export] macro_rules! makro
            ma makro!(…) #[macro_export] macro_rules! makro
            bt u32
        "##]],
    );
}

#[test]
fn bound_for_type_pred() {
    // FIXME: only show traits, macros and modules
    check(
        r#"
struct Foo<'lt, T, const C: usize> where T: $0 {}
"#,
        expect![[r##"
            kw self
            kw super
            kw crate
            tt Trait
            md module
            ma makro!(…) #[macro_export] macro_rules! makro
            ma makro!(…) #[macro_export] macro_rules! makro
        "##]],
    );
}

#[test]
fn bound_for_lifetime_pred() {
    // FIXME: should only show lifetimes here
    check(
        r#"
struct Foo<'lt, T, const C: usize> where 'lt: $0 {}
"#,
        expect![[r##"
            kw self
            kw super
            kw crate
            tt Trait
            md module
            ma makro!(…) #[macro_export] macro_rules! makro
            ma makro!(…) #[macro_export] macro_rules! makro
        "##]],
    );
}

#[test]
fn bound_for_for_pred() {
    check(
        r#"
struct Foo<'lt, T, const C: usize> where for<'a> T: $0 {}
"#,
        expect![[r##"
            kw self
            kw super
            kw crate
            tt Trait
            md module
            ma makro!(…) #[macro_export] macro_rules! makro
            ma makro!(…) #[macro_export] macro_rules! makro
        "##]],
    );
}

#[test]
fn param_list_for_for_pred() {
    check(
        r#"
struct Foo<'lt, T, const C: usize> where for<'a> $0 {}
"#,
        expect![[r##"
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
            ma makro!(…) #[macro_export] macro_rules! makro
            ma makro!(…) #[macro_export] macro_rules! makro
            bt u32
        "##]],
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
        expect![[r##"
            kw self
            kw super
            kw crate
            sp Self
            tt Trait
            en Enum
            st Record
            st Tuple
            ma makro!(…) #[macro_export] macro_rules! makro
            md module
            st Unit
            ma makro!(…) #[macro_export] macro_rules! makro
            bt u32
        "##]],
    );
}
