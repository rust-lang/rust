//! Completion tests for item specifics overall.
//!
//! Except for use items which are tested in [super::use_tree] and mod declarations with are tested
//! in [crate::completions::mod_].
use expect_test::{expect, Expect};

use crate::tests::{completion_list, BASE_ITEMS_FIXTURE};

use super::check_edit;

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{BASE_ITEMS_FIXTURE}{ra_fixture}"));
    expect.assert_eq(&actual)
}

#[test]
fn target_type_or_trait_in_impl_block() {
    check(
        r#"
impl Tra$0
"#,
        expect![[r#"
            en Enum      Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record    Record
            st Tuple     Tuple
            st Unit      Unit
            tt Trait
            un Union     Union
            bt u32       u32
            kw crate::
            kw self::
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
            en Enum      Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record    Record
            st Tuple     Tuple
            st Unit      Unit
            tt Trait
            un Union     Union
            bt u32       u32
            kw crate::
            kw self::
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
    check(
        r"impl Trait f$0",
        expect![[r#"
            kw for
            kw where
        "#]],
    );
    check(
        r"impl Trait for Type $0",
        expect![[r#"
            kw where
        "#]],
    );
}

#[test]
fn completes_where() {
    check(
        r"struct Struct $0",
        expect![[r#"
        kw where
    "#]],
    );
    check(
        r"struct Struct $0 {}",
        expect![[r#"
        kw where
    "#]],
    );
    // FIXME: This shouldn't be completed here
    check(
        r"struct Struct $0 ()",
        expect![[r#"
        kw where
    "#]],
    );
    check(
        r"fn func() $0",
        expect![[r#"
        kw where
    "#]],
    );
    check(
        r"enum Enum $0",
        expect![[r#"
        kw where
    "#]],
    );
    check(
        r"enum Enum $0 {}",
        expect![[r#"
        kw where
    "#]],
    );
    check(
        r"trait Trait $0 {}",
        expect![[r#"
        kw where
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

#[test]
fn add_space_after_vis_kw() {
    check_edit(
        "pub(crate)",
        r"
$0
",
        r#"
pub(crate) $0
"#,
    );

    check_edit(
        "pub",
        r"
$0
",
        r#"
pub $0
"#,
    );

    check_edit(
        "pub(super)",
        r"
$0
",
        r#"
pub(super) $0
"#,
    );

    check_edit(
        "in",
        r"
pub($0)
",
        r#"
pub(in $0)
"#,
    );
}

#[test]
fn add_space_after_unsafe_kw() {
    check_edit(
        "unsafe",
        r"
$0
",
        r#"
unsafe $0
"#,
    );
}

#[test]
fn add_space_after_for_where_kw() {
    check_edit(
        "for",
        r#"
struct S {}

impl Copy $0
"#,
        r#"
struct S {}

impl Copy for $0
"#,
    );

    check_edit(
        "where",
        r#"
struct S {}

impl Copy for S $0
"#,
        r#"
struct S {}

impl Copy for S where $0
"#,
    );
}
