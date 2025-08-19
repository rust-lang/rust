//! Completion tests for item list position.
use expect_test::expect;

use crate::tests::{check, check_edit, check_with_base_items};

#[test]
fn in_mod_item_list() {
    check_with_base_items(
        r#"mod tests { $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            kw async
            kw const
            kw crate::
            kw enum
            kw extern
            kw fn
            kw impl
            kw impl for
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
    check_with_base_items(
        r#"$0"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw async
            kw const
            kw crate::
            kw enum
            kw extern
            kw fn
            kw impl
            kw impl for
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
    check_with_base_items(
        r#"#[attr] $0"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw async
            kw const
            kw crate::
            kw enum
            kw extern
            kw fn
            kw impl
            kw impl for
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
    check_with_base_items(
        r#"crate::$0"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
        "#]],
    )
}

#[test]
fn after_unsafe_token() {
    check_with_base_items(
        r#"unsafe $0"#,
        expect![[r#"
            kw async
            kw extern
            kw fn
            kw impl
            kw impl for
            kw trait
        "#]],
    );
}

#[test]
fn after_async_token() {
    check_with_base_items(
        r#"async $0"#,
        expect![[r#"
            kw fn
            kw unsafe
        "#]],
    );
}

#[test]
fn after_visibility() {
    check_with_base_items(
        r#"pub $0"#,
        expect![[r#"
            kw async
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
    check_with_base_items(
        r#"pub unsafe $0"#,
        expect![[r#"
            kw async
            kw fn
            kw trait
        "#]],
    );
}

#[test]
fn in_impl_assoc_item_list() {
    check_with_base_items(
        r#"impl Struct { $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw async
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
    check_with_base_items(
        r#"impl Struct { #[attr] $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw async
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
    check_with_base_items(
        r"trait Foo { $0 }",
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw async
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
    check_with_base_items(
        r#"trait Foo { fn function(); $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw async
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
    check_with_base_items(
        r#"trait Foo { const CONST: (); $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw async
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
    check_with_base_items(
        r#"trait Foo { type Type; $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw async
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
    check_with_base_items(
        r#"
trait Test {
    type Type0;
    type Type1;
    const CONST0: ();
    const CONST1: ();
    fn function0();
    fn function1();
    async fn function2();
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
            fn async fn function2()
            fn fn function1()
            fn fn function2()
            ma makro!(…) macro_rules! makro
            md module
            ta type Type1 =
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn in_trait_impl_no_unstable_item_on_stable() {
    check(
        r#"
trait Test {
    #[unstable]
    type Type;
    #[unstable]
    const CONST: ();
    #[unstable]
    fn function();
}

impl Test for () {
    $0
}
"#,
        expect![[r#"
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn in_trait_impl_unstable_item_on_nightly() {
    check(
        r#"
//- toolchain:nightly
trait Test {
    #[unstable]
    type Type;
    #[unstable]
    const CONST: ();
    #[unstable]
    fn function();
}

impl Test for () {
    $0
}
"#,
        expect![[r#"
            ct const CONST: () =
            fn fn function()
            ta type Type =
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn after_unit_struct() {
    check_with_base_items(
        r#"struct S; f$0"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw async
            kw const
            kw crate::
            kw enum
            kw extern
            kw fn
            kw impl
            kw impl for
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
type O<'a>
where
Self: 'a;
}
impl B for A {
$0
}
",
        r#"
struct A;
trait B {
type O<'a>
where
Self: 'a;
}
impl B for A {
type O<'a> = $0
where
Self: 'a;
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

#[test]
fn inside_extern_blocks() {
    // Should suggest `fn`, `static`, `unsafe`
    check_with_base_items(
        r#"extern { $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw crate::
            kw fn
            kw pub
            kw pub(crate)
            kw pub(super)
            kw self::
            kw static
            kw unsafe
        "#]],
    );

    // Should suggest `fn`, `static`, `safe`, `unsafe`
    check_with_base_items(
        r#"unsafe extern { $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw crate::
            kw fn
            kw pub
            kw pub(crate)
            kw pub(super)
            kw safe
            kw self::
            kw static
            kw unsafe
        "#]],
    );

    check_with_base_items(
        r#"unsafe extern { pub safe $0 }"#,
        expect![[r#"
            kw fn
            kw static
        "#]],
    );

    check_with_base_items(
        r#"unsafe extern { pub unsafe $0 }"#,
        expect![[r#"
            kw fn
            kw static
        "#]],
    )
}

#[test]
fn tokens_from_macro() {
    check_edit(
        "fn as_ref",
        r#"
//- proc_macros: identity
//- minicore: as_ref
struct Foo;

#[proc_macros::identity]
impl<'a> AsRef<&'a i32> for Foo {
    $0
}
    "#,
        r#"
struct Foo;

#[proc_macros::identity]
impl<'a> AsRef<&'a i32> for Foo {
    fn as_ref(&self) -> &&'a i32 {
    $0
}
}
    "#,
    );
}
