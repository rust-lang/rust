//! Completion tests for item specifics overall.
//!
//! Except for use items which are tested in [super::use_tree] and mod declarations with are tested
//! in [crate::completions::mod_].
use expect_test::{expect, Expect};

use crate::tests::{completion_list, BASE_ITEMS_FIXTURE};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{}{}", BASE_ITEMS_FIXTURE, ra_fixture));
    expect.assert_eq(&actual)
}

#[test]
fn target_type_or_trait_in_impl_block() {
    check(
        r#"
impl Tra$0
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
            kw crate::
            kw self::
            kw super::
        "#]],
    )
}

#[test]
fn target_type_in_trait_impl_block() {
    check(
        r#"
impl Trait for Str$0
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
            kw crate::
            kw self::
            kw super::
        "#]],
    )
}

#[test]
fn after_trait_name_in_trait_def() {
    check(
        r"trait A $0",
        expect![[r#"
            kw where
        "#]],
    );
}

#[test]
fn after_target_name_in_impl() {
    check(
        r"impl Trait $0",
        expect![[r#"
            kw for
            kw where
        "#]],
    );
    // FIXME: This should not emit `kw for`
    check(
        r"impl Trait for Type $0",
        expect![[r#"
            kw for
            kw where
        "#]],
    );
}

#[test]
fn after_struct_name() {
    // FIXME: This should emit `kw where` only
    check(
        r"struct Struct $0",
        expect![[r#"
            ma makro!(…)           macro_rules! makro
            md module
            kw const
            kw crate::
            kw enum
            kw extern
            kw fn
            kw impl
            kw mod
            kw pub
            kw pub(crate)
            kw pub(super)
            kw self::
            kw static
            kw struct
            kw super::
            kw trait
            kw type
            kw union
            kw unsafe
            kw use
            sn macro_rules
            sn tfn (Test function)
            sn tmod (Test module)
        "#]],
    );
}

#[test]
fn after_fn_name() {
    // FIXME: This should emit `kw where` only
    check(
        r"fn func() $0",
        expect![[r#"
            ma makro!(…)           macro_rules! makro
            md module
            kw const
            kw crate::
            kw enum
            kw extern
            kw fn
            kw impl
            kw mod
            kw pub
            kw pub(crate)
            kw pub(super)
            kw self::
            kw static
            kw struct
            kw super::
            kw trait
            kw type
            kw union
            kw unsafe
            kw use
            sn macro_rules
            sn tfn (Test function)
            sn tmod (Test module)
        "#]],
    );
}

#[test]
fn before_record_field() {
    check(
        r#"
struct Foo {
    $0
    pub f: i32,
}
"#,
        expect![[r#"
            kw pub
            kw pub(crate)
            kw pub(super)
        "#]],
    )
}
