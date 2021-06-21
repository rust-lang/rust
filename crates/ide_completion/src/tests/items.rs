//! Completions tests for item specifics overall.
//!
//! Except for use items which are tested in [super::use_tree] and mod declarations with are tested
//! in [crate::completions::mod_].
use expect_test::{expect, Expect};

use crate::tests::completion_list;

fn check(ra_fixture: &str, expect: Expect) {
    let base = r#"#[rustc_builtin_macro]
pub macro Clone {}
enum Enum { Variant }
struct Struct {}
#[macro_export]
macro_rules! foo {}
mod bar {}
const CONST: () = ();
trait Trait {}
"#;
    let actual = completion_list(&format!("{}{}", base, ra_fixture));
    expect.assert_eq(&actual)
}

#[test]
fn target_type_or_trait_in_impl_block() {
    check(
        r#"
impl Tra$0
"#,
        expect![[r##"
            tt Trait
            en Enum
            st Struct
            md bar
            ma foo!(…) #[macro_export] macro_rules! foo
            ma foo!(…) #[macro_export] macro_rules! foo
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
            tt Trait
            en Enum
            st Struct
            md bar
            ma foo!(…) #[macro_export] macro_rules! foo
            ma foo!(…) #[macro_export] macro_rules! foo
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
fn after_trait_or_target_name_in_impl() {
    check(
        r"impl Trait $0",
        expect![[r#"
            kw where
            kw for
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
            kw pub(crate)
            kw pub
        "#]],
    )
}
