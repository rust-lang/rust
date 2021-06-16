use expect_test::expect;

use crate::tests::check;

#[test]
fn in_mod_item_list() {
    check(
        r#"mod tests {
    $0
}
"#,
        expect![[r#"
            kw pub(crate)
            kw pub
            kw unsafe
            kw fn
            kw const
            kw type
            kw use
            kw impl
            kw trait
            kw static
            kw extern
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
fn in_source_file_item_list() {
    check(
        r#"
enum Enum { Variant }
struct MyStruct {}
#[macro_export]
macro_rules! foo {}
mod bar {}
const CONST: () = ();

$0"#,
        expect![[r##"
            kw pub(crate)
            kw pub
            kw unsafe
            kw fn
            kw const
            kw type
            kw use
            kw impl
            kw trait
            kw static
            kw extern
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
fn in_qualified_path() {
    check(
        r#"
enum Enum { Variant }
struct MyStruct {}
#[macro_export]
macro_rules! foo {}
mod bar {}
const CONST: () = ();

crate::$0"#,
        expect![[r##"
            kw pub(crate)
            kw pub
            kw unsafe
            kw fn
            kw const
            kw type
            kw use
            kw impl
            kw trait
            kw static
            kw extern
            kw mod
            kw enum
            kw struct
            kw union
            sn tmod (Test module)
            sn tfn (Test function)
            sn macro_rules
            md bar
            ma foo!(…)          #[macro_export] macro_rules! foo
        "##]],
    )
}

#[test]
fn after_unsafe_token() {
    check(
        r#"
enum Enum { Variant }
struct MyStruct {}
#[macro_export]
macro_rules! foo {}
mod bar {}
const CONST: () = ();

unsafe $0"#,
        expect![[r##"
            kw fn
            kw trait
            kw impl
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
fn after_visibility() {
    check(
        r#"
enum Enum { Variant }
struct MyStruct {}
#[macro_export]
macro_rules! foo {}
mod bar {}
const CONST: () = ();

pub $0"#,
        expect![[r##"
            kw unsafe
            kw fn
            kw const
            kw type
            kw use
            kw impl
            kw trait
            kw static
            kw extern
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
    );
}
