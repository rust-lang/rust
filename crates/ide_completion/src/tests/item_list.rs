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
fn in_mod_item_list() {
    check(
        r#"mod tests { $0 }"#,
        expect![[r##"
            kw pub(crate)
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
            ma foo!(…)          #[macro_export] macro_rules! foo
        "##]],
    )
}

#[test]
fn in_source_file_item_list() {
    check(
        r#"$0"#,
        expect![[r##"
            kw pub(crate)
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
            md bar
            ma foo!(…)          #[macro_export] macro_rules! foo
            ma foo!(…)          #[macro_export] macro_rules! foo
        "##]],
    )
}

#[test]
fn in_item_list_after_attr() {
    check(
        r#"#[attr] $0"#,
        expect![[r#"
            kw pub(crate)
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
        "#]],
    )
}

#[test]
fn in_qualified_path() {
    check(
        r#"crate::$0"#,
        expect![[r##"
            kw pub(crate)
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
            md bar
            ma foo!(…)    #[macro_export] macro_rules! foo
        "##]],
    )
}

#[test]
fn after_unsafe_token() {
    check(
        r#"unsafe $0"#,
        expect![[r#"
            kw fn
            kw trait
            kw impl
        "#]],
    );
}

#[test]
fn after_visibility() {
    check(
        r#"pub $0"#,
        expect![[r#"
            kw unsafe
            kw fn
            kw const
            kw type
            kw use
            kw trait
            kw static
            kw mod
            kw enum
            kw struct
            kw union
        "#]],
    );
}

#[test]
fn after_visibility_unsafe() {
    // FIXME this shouldn't show `impl`
    check(
        r#"pub unsafe $0"#,
        expect![[r#"
            kw fn
            kw trait
            kw impl
        "#]],
    );
}

#[test]
fn in_impl_assoc_item_list() {
    check(
        r#"impl Struct { $0 }"#,
        expect![[r##"
            kw pub(crate)
            kw pub
            kw unsafe
            kw fn
            kw const
            kw type
            md bar
            ma foo!(…)    #[macro_export] macro_rules! foo
            ma foo!(…)    #[macro_export] macro_rules! foo
        "##]],
    )
}

#[test]
fn in_impl_assoc_item_list_after_attr() {
    check(
        r#"impl Struct { #[attr] $0 }"#,
        expect![[r#"
            kw pub(crate)
            kw pub
            kw unsafe
            kw fn
            kw const
            kw type
        "#]],
    )
}

#[test]
fn in_trait_assoc_item_list() {
    check(
        r"trait Foo { $0 }",
        expect![[r##"
            kw unsafe
            kw fn
            kw const
            kw type
            md bar
            ma foo!(…) #[macro_export] macro_rules! foo
            ma foo!(…) #[macro_export] macro_rules! foo
        "##]],
    );
}
