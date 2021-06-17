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
    // FIXME: should not complete `Self`
    check(
        r#"
impl My$0
"#,
        expect![[r##"
            sp Self
            tt Trait
            en Enum
            st Struct
            md bar
            ma foo!(…) #[macro_export] macro_rules! foo
            ma foo!(…) #[macro_export] macro_rules! foo
            bt u32
            bt bool
            bt u8
            bt isize
            bt u16
            bt u64
            bt u128
            bt f32
            bt i128
            bt i16
            bt str
            bt i64
            bt char
            bt f64
            bt i32
            bt i8
            bt usize
        "##]],
    )
}

#[test]
fn after_trait_name_in_trait_def() {
    // FIXME: should only complete `where`
    check(
        r"trait A $0",
        expect![[r##"
            kw where
            sn tmod (Test module)
            sn tfn (Test function)
            sn macro_rules
            md bar
            ma foo!(…)          #[macro_export] macro_rules! foo
            ma foo!(…)          #[macro_export] macro_rules! foo
        "##]],
    );
}

#[test]
fn after_trait_or_target_name_in_impl() {
    // FIXME: should only complete `for` and `where`
    check(
        r"impl A $0",
        expect![[r##"
            kw where
            sn tmod (Test module)
            sn tfn (Test function)
            sn macro_rules
            md bar
            ma foo!(…)          #[macro_export] macro_rules! foo
            ma foo!(…)          #[macro_export] macro_rules! foo
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
            kw pub
        "#]],
    )
}
