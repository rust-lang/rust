//! Completion tests for item list position.
use expect_test::{expect, Expect};

use crate::tests::{check_edit, completion_list, BASE_ITEMS_FIXTURE};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{BASE_ITEMS_FIXTURE}{ra_fixture}"));
    expect.assert_eq(&actual)
}

#[test]
fn in_mod_item_list() {
    check(
        r#"mod tests { $0 }"#,
        expect![[r#"
            ma makro!(…)           macro_rules! makro
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
    )
}

#[test]
fn in_source_file_item_list() {
    check(
        r#"$0"#,
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
            kw trait
            kw type
            kw union
            kw unsafe
            kw use
            sn macro_rules
            sn tfn (Test function)
            sn tmod (Test module)
        "#]],
    )
}

#[test]
fn in_item_list_after_attr() {
    check(
        r#"#[attr] $0"#,
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
            kw trait
            kw type
            kw union
            kw unsafe
            kw use
            sn macro_rules
            sn tfn (Test function)
            sn tmod (Test module)
        "#]],
    )
}

#[test]
fn in_qualified_path() {
    check(
        r#"crate::$0"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
        "#]],
    )
}

#[test]
fn after_unsafe_token() {
    check(
        r#"unsafe $0"#,
        expect![[r#"
            kw fn
            kw impl
            kw trait
        "#]],
    );
}

#[test]
fn after_visibility() {
    check(
        r#"pub $0"#,
        expect![[r#"
            kw const
            kw enum
            kw extern
            kw fn
            kw mod
            kw static
            kw struct
            kw trait
            kw type
            kw union
            kw unsafe
            kw use
        "#]],
    );
}

#[test]
fn after_visibility_unsafe() {
    check(
        r#"pub unsafe $0"#,
        expect![[r#"
            kw fn
            kw trait
        "#]],
    );
}

#[test]
fn in_impl_assoc_item_list() {
    check(
        r#"impl Struct { $0 }"#,
        expect![[r#"
            ma makro!(…)  macro_rules! makro
            md module
            kw const
            kw crate::
            kw fn
            kw pub
            kw pub(crate)
            kw pub(super)
            kw self::
            kw unsafe
        "#]],
    )
}

#[test]
fn in_impl_assoc_item_list_after_attr() {
    check(
        r#"impl Struct { #[attr] $0 }"#,
        expect![[r#"
            ma makro!(…)  macro_rules! makro
            md module
            kw const
            kw crate::
            kw fn
            kw pub
            kw pub(crate)
            kw pub(super)
            kw self::
            kw unsafe
        "#]],
    )
}

#[test]
fn in_trait_assoc_item_list() {
    check(
        r"trait Foo { $0 }",
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw const
            kw crate::
            kw fn
            kw self::
            kw type
            kw unsafe
        "#]],
    );
}

#[test]
fn in_trait_assoc_fn_missing_body() {
    check(
        r#"trait Foo { fn function(); $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw const
            kw crate::
            kw fn
            kw self::
            kw type
            kw unsafe
        "#]],
    );
}

#[test]
fn in_trait_assoc_const_missing_body() {
    check(
        r#"trait Foo { const CONST: (); $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw const
            kw crate::
            kw fn
            kw self::
            kw type
            kw unsafe
        "#]],
    );
}

#[test]
fn in_trait_assoc_type_aliases_missing_ty() {
    check(
        r#"trait Foo { type Type; $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw const
            kw crate::
            kw fn
            kw self::
            kw type
            kw unsafe
        "#]],
    );
}

#[test]
fn in_trait_impl_assoc_item_list() {
    check(
        r#"
trait Test {
    type Type0;
    type Type1;
    const CONST0: ();
    const CONST1: ();
    fn function0();
    fn function1();
}

impl Test for () {
    type Type0 = ();
    const CONST0: () = ();
    fn function0() {}
    $0
}
"#,
        expect![[r#"
            ct const CONST1: () =
            fn fn function1()
            ma makro!(…)          macro_rules! makro
            md module
            ta type Type1 =
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn after_unit_struct() {
    check(
        r#"struct S; f$0"#,
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
fn type_in_impl_trait() {
    check_edit(
        "type O",
        r"
struct A;
trait B {
type O: ?Sized;
}
impl B for A {
$0
}
",
        r#"
struct A;
trait B {
type O: ?Sized;
}
impl B for A {
type O = $0;
}
"#,
    );
    check_edit(
        "type O",
        r"
struct A;
trait B {
type O;
}
impl B for A {
$0
}
",
        r#"
struct A;
trait B {
type O;
}
impl B for A {
type O = $0;
}
"#,
    );
    check_edit(
        "type O",
        r"
struct A;
trait B {
type O: ?Sized = u32;
}
impl B for A {
$0
}
",
        r#"
struct A;
trait B {
type O: ?Sized = u32;
}
impl B for A {
type O = $0;
}
"#,
    );
    check_edit(
        "type O",
        r"
struct A;
trait B {
type O = u32;
}
impl B for A {
$0
}
",
        r"
struct A;
trait B {
type O = u32;
}
impl B for A {
type O = $0;
}
",
    )
}
