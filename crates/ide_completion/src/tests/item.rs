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
        expect![[r##"
            kw self
            kw super
            kw crate
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            ma makro!(…) #[macro_export] macro_rules! makro
            un Union
            bt u32
        "##]],
    )
}

#[test]
fn target_type_in_trait_impl_block() {
    check(
        r#"
impl Trait for Str$0
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
            st Unit
            ma makro!(…) #[macro_export] macro_rules! makro
            un Union
            bt u32
        "##]],
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
            kw where
            kw for
        "#]],
    );
    // FIXME: This should emit `kw where`
    check(r"impl Trait for Type $0", expect![[r#""#]]);
}

#[test]
fn after_struct_name() {
    // FIXME: This should emit `kw where` only
    check(
        r"struct Struct $0",
        expect![[r##"
            kw pub(crate)
            kw pub(super)
            kw pub
            kw unsafe
            kw fn
            kw const
            kw type
            kw impl
            kw extern
            kw use
            kw trait
            kw static
            kw mod
            kw enum
            kw struct
            kw union
            sn tmod (Test module)
            sn tfn (Test function)
            sn macro_rules
            kw self
            kw super
            kw crate
            md module
            ma makro!(…)           #[macro_export] macro_rules! makro
        "##]],
    );
}

#[test]
fn after_fn_name() {
    // FIXME: This should emit `kw where` only
    check(
        r"fn func() $0",
        expect![[r##"
            kw pub(crate)
            kw pub(super)
            kw pub
            kw unsafe
            kw fn
            kw const
            kw type
            kw impl
            kw extern
            kw use
            kw trait
            kw static
            kw mod
            kw enum
            kw struct
            kw union
            sn tmod (Test module)
            sn tfn (Test function)
            sn macro_rules
            kw self
            kw super
            kw crate
            md module
            ma makro!(…)           #[macro_export] macro_rules! makro
        "##]],
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
            kw pub(crate)
            kw pub(super)
            kw pub
        "#]],
    )
}
