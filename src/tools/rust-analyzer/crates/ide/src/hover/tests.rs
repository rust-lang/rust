use expect_test::{Expect, expect};
use ide_db::{FileRange, base_db::SourceDatabase};
use syntax::TextRange;

use crate::{
    HoverConfig, HoverDocFormat, MemoryLayoutHoverConfig, MemoryLayoutHoverRenderKind, fixture,
};

const HOVER_BASE_CONFIG: HoverConfig = HoverConfig {
    links_in_hover: false,
    memory_layout: Some(MemoryLayoutHoverConfig {
        size: Some(MemoryLayoutHoverRenderKind::Both),
        offset: Some(MemoryLayoutHoverRenderKind::Both),
        alignment: Some(MemoryLayoutHoverRenderKind::Both),
        niches: true,
    }),
    documentation: true,
    format: HoverDocFormat::Markdown,
    keywords: true,
    max_trait_assoc_items_count: None,
    max_fields_count: Some(5),
    max_enum_variants_count: Some(5),
    max_subst_ty_len: super::SubstTyLen::Unlimited,
    show_drop_glue: true,
};

fn check_hover_no_result(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    let (analysis, position) = fixture::position(ra_fixture);
    let hover = analysis
        .hover(
            &HoverConfig { links_in_hover: true, ..HOVER_BASE_CONFIG },
            FileRange { file_id: position.file_id, range: TextRange::empty(position.offset) },
        )
        .unwrap();
    assert!(hover.is_none(), "hover not expected but found: {:?}", hover.unwrap());
}

#[track_caller]
fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let (analysis, position) = fixture::position(ra_fixture);
    let hover = analysis
        .hover(
            &HoverConfig { links_in_hover: true, ..HOVER_BASE_CONFIG },
            FileRange { file_id: position.file_id, range: TextRange::empty(position.offset) },
        )
        .unwrap()
        .unwrap();

    let content = analysis.db.file_text(position.file_id);
    let hovered_element = &content.text(&analysis.db)[hover.range];

    let actual = format!("*{hovered_element}*\n{}\n", hover.info.markup);
    expect.assert_eq(&actual)
}

#[track_caller]
fn check_hover_fields_limit(
    fields_count: impl Into<Option<usize>>,
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    expect: Expect,
) {
    let (analysis, position) = fixture::position(ra_fixture);
    let hover = analysis
        .hover(
            &HoverConfig {
                links_in_hover: true,
                max_fields_count: fields_count.into(),
                ..HOVER_BASE_CONFIG
            },
            FileRange { file_id: position.file_id, range: TextRange::empty(position.offset) },
        )
        .unwrap()
        .unwrap();

    let content = analysis.db.file_text(position.file_id).text(&analysis.db);
    let hovered_element = &content[hover.range];

    let actual = format!("*{hovered_element}*\n{}\n", hover.info.markup);
    expect.assert_eq(&actual)
}

#[track_caller]
fn check_hover_enum_variants_limit(
    variants_count: impl Into<Option<usize>>,
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    expect: Expect,
) {
    let (analysis, position) = fixture::position(ra_fixture);
    let hover = analysis
        .hover(
            &HoverConfig {
                links_in_hover: true,
                max_enum_variants_count: variants_count.into(),
                ..HOVER_BASE_CONFIG
            },
            FileRange { file_id: position.file_id, range: TextRange::empty(position.offset) },
        )
        .unwrap()
        .unwrap();

    let content = analysis.db.file_text(position.file_id).text(&analysis.db);
    let hovered_element = &content[hover.range];

    let actual = format!("*{hovered_element}*\n{}\n", hover.info.markup);
    expect.assert_eq(&actual)
}

#[track_caller]
fn check_assoc_count(
    count: usize,
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    expect: Expect,
) {
    let (analysis, position) = fixture::position(ra_fixture);
    let hover = analysis
        .hover(
            &HoverConfig {
                links_in_hover: true,
                max_trait_assoc_items_count: Some(count),
                ..HOVER_BASE_CONFIG
            },
            FileRange { file_id: position.file_id, range: TextRange::empty(position.offset) },
        )
        .unwrap()
        .unwrap();

    let content = analysis.db.file_text(position.file_id).text(&analysis.db);
    let hovered_element = &content[hover.range];

    let actual = format!("*{hovered_element}*\n{}\n", hover.info.markup);
    expect.assert_eq(&actual)
}

fn check_hover_no_links(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let (analysis, position) = fixture::position(ra_fixture);
    let hover = analysis
        .hover(
            &HOVER_BASE_CONFIG,
            FileRange { file_id: position.file_id, range: TextRange::empty(position.offset) },
        )
        .unwrap()
        .unwrap();

    let content = analysis.db.file_text(position.file_id).text(&analysis.db);
    let hovered_element = &content[hover.range];

    let actual = format!("*{hovered_element}*\n{}\n", hover.info.markup);
    expect.assert_eq(&actual)
}

fn check_hover_no_memory_layout(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let (analysis, position) = fixture::position(ra_fixture);
    let hover = analysis
        .hover(
            &HoverConfig { memory_layout: None, ..HOVER_BASE_CONFIG },
            FileRange { file_id: position.file_id, range: TextRange::empty(position.offset) },
        )
        .unwrap()
        .unwrap();

    let content = analysis.db.file_text(position.file_id).text(&analysis.db);
    let hovered_element = &content[hover.range];

    let actual = format!("*{hovered_element}*\n{}\n", hover.info.markup);
    expect.assert_eq(&actual)
}

fn check_hover_no_markdown(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let (analysis, position) = fixture::position(ra_fixture);
    let hover = analysis
        .hover(
            &HoverConfig {
                links_in_hover: true,
                format: HoverDocFormat::PlainText,
                ..HOVER_BASE_CONFIG
            },
            FileRange { file_id: position.file_id, range: TextRange::empty(position.offset) },
        )
        .unwrap()
        .unwrap();

    let content = analysis.db.file_text(position.file_id).text(&analysis.db);
    let hovered_element = &content[hover.range];

    let actual = format!("*{hovered_element}*\n{}\n", hover.info.markup);
    expect.assert_eq(&actual)
}

fn check_actions(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let (analysis, file_id, position) = fixture::range_or_position(ra_fixture);
    let mut hover = analysis
        .hover(
            &HoverConfig { links_in_hover: true, ..HOVER_BASE_CONFIG },
            FileRange { file_id, range: position.range_or_empty() },
        )
        .unwrap()
        .unwrap();
    // stub out ranges into minicore as they can change every now and then
    hover.info.actions.iter_mut().for_each(|action| match action {
        super::HoverAction::GoToType(act) => act.iter_mut().for_each(|data| {
            if data.nav.file_id == file_id {
                return;
            }
            data.nav.full_range = TextRange::empty(span::TextSize::new(!0));
            if let Some(range) = &mut data.nav.focus_range {
                *range = TextRange::empty(span::TextSize::new(!0));
            }
        }),
        _ => (),
    });
    expect.assert_debug_eq(&hover.info.actions)
}

fn check_hover_range(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let (analysis, range) = fixture::range(ra_fixture);
    let hover = analysis.hover(&HOVER_BASE_CONFIG, range).unwrap().unwrap();
    expect.assert_eq(hover.info.markup.as_str())
}

fn check_hover_range_actions(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let (analysis, range) = fixture::range(ra_fixture);
    let mut hover = analysis
        .hover(&HoverConfig { links_in_hover: true, ..HOVER_BASE_CONFIG }, range)
        .unwrap()
        .unwrap();
    // stub out ranges into minicore as they can change every now and then
    hover.info.actions.iter_mut().for_each(|action| match action {
        super::HoverAction::GoToType(act) => act.iter_mut().for_each(|data| {
            if data.nav.file_id == range.file_id {
                return;
            }
            data.nav.full_range = TextRange::empty(span::TextSize::new(!0));
            if let Some(range) = &mut data.nav.focus_range {
                *range = TextRange::empty(span::TextSize::new(!0));
            }
        }),
        _ => (),
    });
    expect.assert_debug_eq(&hover.info.actions);
}

fn check_hover_range_no_results(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    let (analysis, range) = fixture::range(ra_fixture);
    let hover = analysis.hover(&HOVER_BASE_CONFIG, range).unwrap();
    assert!(hover.is_none());
}

#[test]
fn hover_descend_macros_avoids_duplicates() {
    check(
        r#"
macro_rules! dupe_use {
    ($local:ident) => {
        {
            $local;
            $local;
        }
    }
}
fn foo() {
    let local = 0;
    dupe_use!(local$0);
}
"#,
        expect![[r#"
            *local*

            ```rust
            let local: i32
            ```
        "#]],
    );
}

#[test]
fn hover_shows_all_macro_descends() {
    check(
        r#"
macro_rules! m {
    ($name:ident) => {
        /// Outer
        fn $name() {}

        mod module {
            /// Inner
            fn $name() {}
        }
    };
}

m!(ab$0c);
            "#,
        expect![[r#"
            *abc*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn abc()
            ```

            ---

            Outer

            ---

            ```rust
            ra_test_fixture::module
            ```

            ```rust
            fn abc()
            ```

            ---

            Inner
        "#]],
    );
}

#[test]
fn hover_remove_markdown_if_configured() {
    check_hover_no_markdown(
        r#"
pub fn foo() -> u32 { 1 }

fn main() {
    let foo_test = foo$0();
}
"#,
        expect![[r#"
            *foo*
            ra_test_fixture

            pub fn foo() -> u32
        "#]],
    );
}

#[test]
fn hover_closure() {
    check(
        r#"
//- minicore: copy
fn main() {
    let x = 2;
    let y = $0|z| x + z;
}
"#,
        expect![[r#"
            *|*
            ```rust
            impl Fn(i32) -> i32
            ```
            ___
            size = 8, align = 8, niches = 1

            ## Captures
            * `x` by immutable borrow
        "#]],
    );

    check(
        r#"
//- minicore: copy
fn foo(x: impl Fn(i32) -> i32) {

}
fn main() {
    foo($0|x: i32| x)
}
"#,
        expect![[r#"
            *|*
            ```rust
            impl Fn(i32) -> i32
            ```
            ___
            size = 0, align = 1

            ## Captures
            This closure captures nothing
        "#]],
    );

    check(
        r#"
//- minicore: copy

struct Z { f: i32 }

struct Y(&'static mut Z)

struct X {
    f1: Y,
    f2: (Y, Y),
}

fn main() {
    let x: X;
    let y = $0|| {
        x.f1;
        &mut x.f2.0 .0.f;
    };
}
"#,
        expect![[r#"
            *|*
            ```rust
            impl FnOnce()
            ```
            ___
            size = 16 (0x10), align = 8, niches = 1

            ## Captures
            * `x.f1` by move
            * `(*x.f2.0.0).f` by mutable borrow
        "#]],
    );
    check(
        r#"
//- minicore: copy, option

fn do_char(c: char) {}

fn main() {
    let x = None;
    let y = |$0| {
        match x {
            Some(c) => do_char(c),
            None => x = None,
        }
    };
}
"#,
        expect![[r#"
            *|*
            ```rust
            impl FnMut()
            ```
            ___
            size = 8, align = 8, niches = 1

            ## Captures
            * `x` by mutable borrow
        "#]],
    );
}

#[test]
fn hover_ranged_closure() {
    check_hover_range(
        r#"
//- minicore: fn
struct S;
struct S2;
fn main() {
    let x = &S;
    let y = ($0|| {x; S2}$0).call();
}
"#,
        expect![[r#"
            ```rust
            impl FnOnce() -> S2
            ```
            ___
            size = 8, align = 8, niches = 1
            Coerced to: &impl FnOnce() -> S2

            ## Captures
            * `x` by move"#]],
    );
    check_hover_range_actions(
        r#"
//- minicore: fn
struct S;
struct S2;
fn main() {
    let x = &S;
    let y = ($0|| {x; S2}$0).call();
}
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S2",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 10..20,
                                focus_range: 17..19,
                                name: "S2",
                                kind: Struct,
                                description: "struct S2",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..9,
                                focus_range: 7..8,
                                name: "S",
                                kind: Struct,
                                description: "struct S",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "core::ops::function::FnOnce",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 4294967295..4294967295,
                                focus_range: 4294967295..4294967295,
                                name: "FnOnce",
                                kind: Trait,
                                container_name: "function",
                                description: "pub trait FnOnce<Args>\nwhere\n    Args: Tuple,",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn hover_shows_long_type_of_an_expression() {
    check(
        r#"
struct Scan<A, B, C> { a: A, b: B, c: C }
struct Iter<I> { inner: I }
enum Option<T> { Some(T), None }

struct OtherStruct<T> { i: T }

fn scan<A, B, C>(a: A, b: B, c: C) -> Iter<Scan<OtherStruct<A>, B, C>> {
    Iter { inner: Scan { a, b, c } }
}

fn main() {
    let num: i32 = 55;
    let closure = |memo: &mut u32, value: &u32, _another: &mut u32| -> Option<u32> {
        Option::Some(*memo + value)
    };
    let number = 5u32;
    let mut iter$0 = scan(OtherStruct { i: num }, closure, number);
}
"#,
        expect![[r#"
            *iter*

            ```rust
            let mut iter: Iter<Scan<OtherStruct<OtherStruct<i32>>, impl Fn(&mut u32, &u32, &mut u32) -> Option<u32>, u32>>
            ```

            ---

            size = 8, align = 4, no Drop
        "#]],
    );
}

#[test]
fn hover_shows_fn_signature() {
    // Single file with result
    check(
        r#"
pub fn foo() -> u32 { 1 }

fn main() { let foo_test = fo$0o(); }
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub fn foo() -> u32
            ```
        "#]],
    );

    // Use literal `crate` in path
    check(
        r#"
pub struct X;

fn foo() -> crate::X { X }

fn main() { f$0oo(); }
        "#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn foo() -> crate::X
            ```
        "#]],
    );

    // Check `super` in path
    check(
        r#"
pub struct X;

mod m { pub fn foo() -> super::X { super::X } }

fn main() { m::f$0oo(); }
        "#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::m
            ```

            ```rust
            pub fn foo() -> super::X
            ```
        "#]],
    );
}

#[test]
fn hover_omits_unnamed_where_preds() {
    check(
        r#"
pub fn foo(bar: impl T) { }

fn main() { fo$0o(); }
        "#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub fn foo(bar: impl T)
            ```
        "#]],
    );
    check(
        r#"
pub fn foo<V: AsRef<str>>(bar: impl T, baz: V) { }

fn main() { fo$0o(); }
        "#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub fn foo<V>(bar: impl T, baz: V)
            where
                V: AsRef<str>,
            ```
        "#]],
    );
}

#[test]
fn hover_shows_fn_signature_with_type_params() {
    check(
        r#"
pub fn foo<'a, T: AsRef<str>>(b: &'a T) -> &'a str { }

fn main() { let foo_test = fo$0o(); }
        "#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub fn foo<'a, T>(b: &'a T) -> &'a str
            where
                T: AsRef<str>,
            ```
        "#]],
    );
}

#[test]
fn hover_shows_fn_signature_on_fn_name() {
    check(
        r#"
pub fn foo$0(a: u32, b: u32) -> u32 {}

fn main() { }
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub fn foo(a: u32, b: u32) -> u32
            ```
        "#]],
    );
}

#[test]
fn hover_shows_fn_doc() {
    check(
        r#"
/// # Example
/// ```
/// # use std::path::Path;
/// #
/// foo(Path::new("hello, world!"))
/// ```
pub fn foo$0(_: &Path) {}

fn main() { }
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub fn foo(_: &Path)
            ```

            ---

            # Example

            ```
            # use std::path::Path;
            #
            foo(Path::new("hello, world!"))
            ```
        "#]],
    );
}

#[test]
fn hover_shows_fn_doc_attr_raw_string() {
    check(
        r##"
#[doc = r#"Raw string doc attr"#]
pub fn foo$0(_: &Path) {}

fn main() { }
"##,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub fn foo(_: &Path)
            ```

            ---

            Raw string doc attr
        "#]],
    );
}

#[test]
fn hover_field_offset() {
    // Hovering over the field when instantiating
    check(
        r#"
struct Foo { fiel$0d_a: u8, field_b: i32, field_c: i16 }
"#,
        expect![[r#"
            *field_a*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            field_a: u8
            ```

            ---

            size = 1, align = 1, offset = 6, no Drop
        "#]],
    );
}

#[test]
fn hover_shows_struct_field_info() {
    // Hovering over the field when instantiating
    check(
        r#"
struct Foo { pub field_a: u32 }

fn main() {
    let foo = Foo { field_a$0: 0, };
}
"#,
        expect![[r#"
            *field_a*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            pub field_a: u32
            ```
        "#]],
    );

    // Hovering over the field in the definition
    check(
        r#"
struct Foo { pub field_a$0: u32 }

fn main() {
    let foo = Foo { field_a: 0 };
}
"#,
        expect![[r#"
            *field_a*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            pub field_a: u32
            ```

            ---

            size = 4, align = 4, offset = 0, no Drop
        "#]],
    );
}

#[test]
fn hover_shows_tuple_struct_field_info() {
    check(
        r#"
struct Foo(pub u32)

fn main() {
    let foo = Foo { 0$0: 0, };
}
"#,
        expect![[r#"
            *0*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            pub 0: u32
            ```
        "#]],
    );
    check(
        r#"
struct Foo(pub u32)

fn foo(foo: Foo) {
    foo.0$0;
}
"#,
        expect![[r#"
            *0*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            pub 0: u32
            ```
        "#]],
    );
}

#[test]
fn hover_tuple_struct() {
    check(
        r#"
struct Foo$0(pub u32) where u32: Copy;
"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo(pub u32)
            where
                u32: Copy,
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
}

#[test]
fn hover_record_struct() {
    check(
        r#"
struct Foo$0 { field: u32 }
"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo {
                field: u32,
            }
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
    check(
        r#"
struct Foo$0 where u32: Copy { field: u32 }
"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo
            where
                u32: Copy,
            {
                field: u32,
            }
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
}

#[test]
fn hover_record_struct_limit() {
    check_hover_fields_limit(
        3,
        r#"
    struct Foo$0 { a: u32, b: i32, c: i32 }
    "#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo {
                a: u32,
                b: i32,
                c: i32,
            }
            ```

            ---

            size = 12 (0xC), align = 4, no Drop
        "#]],
    );
    check_hover_fields_limit(
        3,
        r#"
    struct Foo$0 { a: u32 }
    "#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo {
                a: u32,
            }
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
    check_hover_fields_limit(
        3,
        r#"
    struct Foo$0 { a: u32, b: i32, c: i32, d: u32 }
    "#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo {
                a: u32,
                b: i32,
                c: i32,
                /* … */
            }
            ```

            ---

            size = 16 (0x10), align = 4, no Drop
        "#]],
    );
    check_hover_fields_limit(
        None,
        r#"
    struct Foo$0 { a: u32, b: i32, c: i32 }
    "#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo
            ```

            ---

            size = 12 (0xC), align = 4, no Drop
        "#]],
    );
    check_hover_fields_limit(
        0,
        r#"
    struct Foo$0 { a: u32, b: i32, c: i32 }
    "#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo { /* … */ }
            ```

            ---

            size = 12 (0xC), align = 4, no Drop
        "#]],
    );

    // No extra spaces within `{}` when there are no fields
    check_hover_fields_limit(
        5,
        r#"
    struct Foo$0 {}
    "#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo {}
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn hover_record_variant_limit() {
    check_hover_fields_limit(
        3,
        r#"
    enum Foo { A$0 { a: u32, b: i32, c: i32 } }
    "#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            A { a: u32, b: i32, c: i32, }
            ```

            ---

            size = 12 (0xC), align = 4, no Drop
        "#]],
    );
    check_hover_fields_limit(
        3,
        r#"
    enum Foo { A$0 { a: u32 } }
    "#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            A { a: u32, }
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
    check_hover_fields_limit(
        3,
        r#"
    enum Foo { A$0 { a: u32, b: i32, c: i32, d: u32 } }
    "#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            A { a: u32, b: i32, c: i32, /* … */ }
            ```

            ---

            size = 16 (0x10), align = 4, no Drop
        "#]],
    );
    check_hover_fields_limit(
        None,
        r#"
    enum Foo { A$0 { a: u32, b: i32, c: i32 } }
    "#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            A
            ```

            ---

            size = 12 (0xC), align = 4, no Drop
        "#]],
    );
    check_hover_fields_limit(
        0,
        r#"
    enum Foo { A$0 { a: u32, b: i32, c: i32 } }
    "#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            A { /* … */ }
            ```

            ---

            size = 12 (0xC), align = 4, no Drop
        "#]],
    );
}

#[test]
fn hover_enum_limit() {
    check_hover_enum_variants_limit(
        5,
        r#"enum Foo$0 { A, B }"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            enum Foo {
                A,
                B,
            }
            ```

            ---

            size = 1, align = 1, niches = 254, no Drop
        "#]],
    );
    check_hover_enum_variants_limit(
        1,
        r#"enum Foo$0 { A, B }"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            enum Foo {
                A,
                /* … */
            }
            ```

            ---

            size = 1, align = 1, niches = 254, no Drop
        "#]],
    );
    check_hover_enum_variants_limit(
        0,
        r#"enum Foo$0 { A, B }"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            enum Foo { /* … */ }
            ```

            ---

            size = 1, align = 1, niches = 254, no Drop
        "#]],
    );
    check_hover_enum_variants_limit(
        None,
        r#"enum Foo$0 { A, B }"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            enum Foo
            ```

            ---

            size = 1, align = 1, niches = 254, no Drop
        "#]],
    );
    check_hover_enum_variants_limit(
        7,
        r#"enum Enum$0 {
               Variant {},
               Variant2 { field: i32 },
               Variant3 { field: i32, field2: i32 },
               Variant4(),
               Variant5(i32),
               Variant6(i32, i32),
               Variant7,
               Variant8,
           }"#,
        expect![[r#"
            *Enum*

            ```rust
            ra_test_fixture
            ```

            ```rust
            enum Enum {
                Variant {},
                Variant2 { /* … */ },
                Variant3 { /* … */ },
                Variant4(),
                Variant5( /* … */ ),
                Variant6( /* … */ ),
                Variant7,
                /* … */
            }
            ```

            ---

            size = 12 (0xC), align = 4, niches = a lot, no Drop
        "#]],
    );
}

#[test]
fn hover_union_limit() {
    check_hover_fields_limit(
        5,
        r#"union Foo$0 { a: u32, b: i32 }"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            union Foo {
                a: u32,
                b: i32,
            }
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
    check_hover_fields_limit(
        1,
        r#"union Foo$0 { a: u32, b: i32 }"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            union Foo {
                a: u32,
                /* … */
            }
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
    check_hover_fields_limit(
        0,
        r#"union Foo$0 { a: u32, b: i32 }"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            union Foo { /* … */ }
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
    check_hover_fields_limit(
        None,
        r#"union Foo$0 { a: u32, b: i32 }"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            union Foo
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
}

#[test]
fn hover_unit_struct() {
    check(
        r#"
struct Foo$0 where u32: Copy;
"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo
            where
                u32: Copy,
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn hover_type_alias() {
    check(
        r#"
type Fo$0o: Trait = S where T: Trait;
"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            type Foo: Trait = S
            where
                T: Trait,
            ```

            ---

            no Drop
        "#]],
    );
}

#[test]
fn hover_const_static() {
    check(
        r#"const foo$0: u32 = 123;"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const foo: u32 = 123 (0x7B)
            ```
        "#]],
    );
    check(
        r#"
const foo$0: u32 = {
    let x = foo();
    x + 100
};"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const foo: u32 = {
                let x = foo();
                x + 100
            }
            ```
        "#]],
    );

    check(
        r#"static foo$0: u32 = 456;"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            static foo: u32 = 456 (0x1C8)
            ```
        "#]],
    );

    check(
        r#"const FOO$0: i32 = -2147483648;"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: i32 = -2147483648 (0x80000000)
            ```
        "#]],
    );

    check(
        r#"
        const FOO: i32 = -2147483648;
        const BAR$0: bool = FOO > 0;
        "#,
        expect![[r#"
            *BAR*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const BAR: bool = false
            ```
        "#]],
    );
}

#[test]
fn hover_unsigned_max_const() {
    check(
        r#"const $0A: u128 = -1_i128 as u128;"#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const A: u128 = 340282366920938463463374607431768211455 (0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)
            ```
        "#]],
    );
}

#[test]
fn hover_eval_complex_constants() {
    check(
        r#"
        struct X { f1: (), f2: i32 }
        const foo$0: (i8, X, i64) = (1, X { f2: 5 - 1, f1: () }, 1 - 2);
        "#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const foo: (i8, X, i64) = (1, X { f1: (), f2: 4 }, -1)
            ```
        "#]],
    );
}

#[test]
fn hover_default_generic_types() {
    check(
        r#"
struct Test<K, T = u8> { k: K, t: T }

fn main() {
    let zz$0 = Test { t: 23u8, k: 33 };
}"#,
        expect![[r#"
            *zz*

            ```rust
            let zz: Test<i32>
            ```

            ---

            size = 8, align = 4, no Drop
        "#]],
    );
    check_hover_range(
        r#"
struct Test<K, T = u8> { k: K, t: T }

fn main() {
    let $0zz$0 = Test { t: 23u8, k: 33 };
}"#,
        expect![[r#"
            ```rust
            Test<i32, u8>
            ```"#]],
    );
}

#[test]
fn hover_some() {
    check(
        r#"
enum Option<T> { Some(T) }
use Option::Some;

fn main() { So$0me(12); }
"#,
        expect![[r#"
            *Some*

            ```rust
            ra_test_fixture::Option
            ```

            ```rust
            Some(T)
            ```
        "#]],
    );

    check(
        r#"
enum Option<T> { Some(T) }
use Option::Some;

fn main() { let b$0ar = Some(12); }
"#,
        expect![[r#"
            *bar*

            ```rust
            let bar: Option<i32>
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
}

#[test]
fn hover_enum_variant() {
    check(
        r#"
enum Option<T> {
    Some(T)
    /// The None variant
    Non$0e
}
"#,
        expect![[r#"
            *None*

            ```rust
            ra_test_fixture::Option
            ```

            ```rust
            None
            ```

            ---

            no Drop

            ---

            The None variant
        "#]],
    );

    check(
        r#"
enum Option<T> {
    /// The Some variant
    Some(T)
}
fn main() {
    let s = Option::Som$0e(12);
}
"#,
        expect![[r#"
            *Some*

            ```rust
            ra_test_fixture::Option
            ```

            ```rust
            Some(T)
            ```

            ---

            The Some variant
        "#]],
    );
}

#[test]
fn hover_for_local_variable() {
    check(
        r#"fn func(foo: i32) { fo$0o; }"#,
        expect![[r#"
            *foo*

            ```rust
            foo: i32
            ```
        "#]],
    )
}

#[test]
fn hover_for_local_variable_pat() {
    check(
        r#"fn func(fo$0o: i32) {}"#,
        expect![[r#"
            *foo*

            ```rust
            foo: i32
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    )
}

#[test]
fn hover_local_var_edge() {
    check(
        r#"fn func(foo: i32) { if true { $0foo; }; }"#,
        expect![[r#"
            *foo*

            ```rust
            foo: i32
            ```
        "#]],
    )
}

#[test]
fn hover_for_param_edge() {
    check(
        r#"fn func($0foo: i32) {}"#,
        expect![[r#"
            *foo*

            ```rust
            foo: i32
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    )
}

#[test]
fn hover_for_param_with_multiple_traits() {
    check(
        r#"
            //- minicore: sized
            trait Deref {
                type Target: ?Sized;
            }
            trait DerefMut {
                type Target: ?Sized;
            }
            fn f(_x$0: impl Deref<Target=u8> + DerefMut<Target=u8>) {}"#,
        expect![[r#"
            *_x*

            ```rust
            _x: impl Deref<Target = u8> + DerefMut<Target = u8>
            ```

            ---

            type param may need Drop
        "#]],
    )
}

#[test]
fn test_hover_infer_associated_method_result() {
    check(
        r#"
struct Thing { x: u32 }

impl Thing {
    fn new() -> Thing { Thing { x: 0 } }
}

fn main() { let foo_$0test = Thing::new(); }
"#,
        expect![[r#"
            *foo_test*

            ```rust
            let foo_test: Thing
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    )
}

#[test]
fn test_hover_infer_associated_method_exact() {
    check(
        r#"
mod wrapper {
    pub struct Thing { x: u32 }

    impl Thing {
        pub fn new() -> Thing { Thing { x: 0 } }
    }
}

fn main() { let foo_test = wrapper::Thing::new$0(); }
"#,
        expect![[r#"
            *new*

            ```rust
            ra_test_fixture::wrapper::Thing
            ```

            ```rust
            pub fn new() -> Thing
            ```
        "#]],
    )
}

#[test]
fn test_hover_infer_associated_const_in_pattern() {
    check(
        r#"
struct X;
impl X {
    const C: u32 = 1;
}

fn main() {
    match 1 {
        X::C$0 => {},
        2 => {},
        _ => {}
    };
}
"#,
        expect![[r#"
            *C*

            ```rust
            ra_test_fixture::X
            ```

            ```rust
            const C: u32 = 1
            ```
        "#]],
    )
}

#[test]
fn test_hover_self() {
    check(
        r#"
struct Thing { x: u32 }
impl Thing {
    fn new() -> Self { Self$0 { x: 0 } }
}
"#,
        expect![[r#"
            *Self*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Thing {
                x: u32,
            }
            ```

            ---

            size = 4, align = 4
        "#]],
    );
    check_hover_fields_limit(
        None,
        r#"
struct Thing { x: u32 }
impl Thing {
    fn new() -> Self$0 { Self { x: 0 } }
}
"#,
        expect![[r#"
            *Self*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Thing
            ```

            ---

            size = 4, align = 4
        "#]],
    );
    check(
        r#"
struct Thing { x: u32 }
impl Thing {
    fn new() -> Self$0 { Self { x: 0 } }
}
"#,
        expect![[r#"
            *Self*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Thing {
                x: u32,
            }
            ```

            ---

            size = 4, align = 4
        "#]],
    );
    check(
        r#"
enum Thing { A }
impl Thing {
    pub fn new() -> Self$0 { Thing::A }
}
"#,
        expect![[r#"
            *Self*

            ```rust
            ra_test_fixture
            ```

            ```rust
            enum Thing {
                A,
            }
            ```

            ---

            size = 0, align = 1
        "#]],
    );
    check(
        r#"
enum Thing { A }
impl Thing {
    pub fn thing(a: Self$0) {}
}
"#,
        expect![[r#"
            *Self*

            ```rust
            ra_test_fixture
            ```

            ```rust
            enum Thing {
                A,
            }
            ```

            ---

            size = 0, align = 1
        "#]],
    );
    check(
        r#"
impl usize {
    pub fn thing(a: Self$0) {}
}
"#,
        expect![[r#"
            *Self*

            ```rust
            ra_test_fixture
            ```

            ```rust
            usize
            ```

            ---

            size = 8, align = 8
        "#]],
    );
    check(
        r#"
impl fn() -> usize {
    pub fn thing(a: Self$0) {}
}
"#,
        expect![[r#"
            *Self*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn() -> usize
            ```

            ---

            size = 8, align = 8, niches = 1
        "#]],
    );
    check(
        r#"
pub struct Foo
where
    Self$0:;
"#,
        expect![[r#"
            *Self*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub struct Foo
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn test_hover_shadowing_pat() {
    check(
        r#"
fn x() {}

fn y() {
    let x = 0i32;
    x$0;
}
"#,
        expect![[r#"
            *x*

            ```rust
            let x: i32
            ```
        "#]],
    )
}

#[test]
fn test_hover_macro_invocation() {
    check(
        r#"
macro_rules! foo { (a) => {}; () => {} }

fn f() { fo$0o!(); }
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            macro_rules! foo // matched arm #1
            ```
        "#]],
    )
}

#[test]
fn test_hover_macro2_invocation() {
    check(
        r#"
/// foo bar
///
/// foo bar baz
macro foo() {}

fn f() { fo$0o!(); }
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            macro foo // matched arm #0
            ```

            ---

            foo bar

            foo bar baz
        "#]],
    )
}

#[test]
fn test_hover_tuple_field() {
    check(
        r#"struct TS(String, i32$0);"#,
        expect![[r#"
            *i32*

            ```rust
            i32
            ```
        "#]],
    )
}

#[test]
fn test_hover_through_macro() {
    check(
        r#"
macro_rules! id { ($($tt:tt)*) => { $($tt)* } }
fn foo() {}
id! {
    fn bar() { fo$0o(); }
}
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn foo()
            ```
        "#]],
    );
}

#[test]
fn test_hover_through_attr() {
    check(
        r#"
//- proc_macros: identity
#[proc_macros::identity]
fn foo$0() {}
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn foo()
            ```
        "#]],
    );
}

#[test]
fn test_hover_through_expr_in_macro() {
    check(
        r#"
macro_rules! id { ($($tt:tt)*) => { $($tt)* } }
fn foo(bar:u32) { let a = id!(ba$0r); }
"#,
        expect![[r#"
            *bar*

            ```rust
            bar: u32
            ```
        "#]],
    );
}

#[test]
fn test_hover_through_expr_in_macro_recursive() {
    check(
        r#"
macro_rules! id_deep { ($($tt:tt)*) => { $($tt)* } }
macro_rules! id { ($($tt:tt)*) => { id_deep!($($tt)*) } }
fn foo(bar:u32) { let a = id!(ba$0r); }
"#,
        expect![[r#"
            *bar*

            ```rust
            bar: u32
            ```
        "#]],
    );
}

#[test]
fn test_hover_through_func_in_macro_recursive() {
    check(
        r#"
macro_rules! id_deep { ($($tt:tt)*) => { $($tt)* } }
macro_rules! id { ($($tt:tt)*) => { id_deep!($($tt)*) } }
fn bar() -> u32 { 0 }
fn foo() { let a = id!([0u32, bar$0()] ); }
"#,
        expect![[r#"
            *bar*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn bar() -> u32
            ```
        "#]],
    );
}

#[test]
fn test_hover_through_assert_macro() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! assert {}

fn bar() -> bool { true }
fn foo() {
    assert!(ba$0r());
}
"#,
        expect![[r#"
            *bar*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn bar() -> bool
            ```
        "#]],
    );
}

#[test]
fn test_hover_multiple_actions() {
    check_actions(
        r#"
struct Bar;
struct Foo { bar: Bar }

fn foo(Foo { b$0ar }: &Foo) {}
        "#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Bar",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..11,
                                focus_range: 7..10,
                                name: "Bar",
                                kind: Struct,
                                description: "struct Bar",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    )
}

#[test]
fn test_hover_show_type_def_for_subst() {
    check_actions(
        r#"
fn f<T>(t: T) {

}

struct S;

fn test() {
    let a = S;
    f$0(a);
}
"#,
        expect![[r#"
            [
                Reference(
                    FilePositionWrapper {
                        file_id: FileId(
                            0,
                        ),
                        offset: 3,
                    },
                ),
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 20..29,
                                focus_range: 27..28,
                                name: "S",
                                kind: Struct,
                                description: "struct S",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_show_type_def_for_func_param() {
    check_actions(
        r#"
struct Bar;
fn f(b: Bar) {

}

fn test() {
    let b = Bar;
    f$0(b);
}
"#,
        expect![[r#"
            [
                Reference(
                    FilePositionWrapper {
                        file_id: FileId(
                            0,
                        ),
                        offset: 15,
                    },
                ),
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Bar",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..11,
                                focus_range: 7..10,
                                name: "Bar",
                                kind: Struct,
                                description: "struct Bar",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_show_type_def_for_trait_bound() {
    check_actions(
        r#"
trait Bar {}
fn f<T: Bar>(b: T) {

}

fn test() {
    f$0();
}
"#,
        expect![[r#"
            [
                Reference(
                    FilePositionWrapper {
                        file_id: FileId(
                            0,
                        ),
                        offset: 16,
                    },
                ),
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Bar",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..12,
                                focus_range: 6..9,
                                name: "Bar",
                                kind: Trait,
                                description: "trait Bar",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_non_ascii_space_doc() {
    check(
        "
///　<- `\u{3000}` here
fn foo() { }

fn bar() { fo$0o(); }
",
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn foo()
            ```

            ---

            \<- `　` here
        "#]],
    );
}

#[test]
fn test_hover_function_show_qualifiers() {
    check(
        r#"async fn foo$0() {}"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            async fn foo()
            ```
        "#]],
    );
    check(
        r#"pub const unsafe fn foo$0() {}"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub const unsafe fn foo()
            ```
        "#]],
    );
    // Top level `pub(crate)` will be displayed as no visibility.
    check(
        r#"mod m { pub(crate) async unsafe extern "C" fn foo$0() {} }"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::m
            ```

            ```rust
            pub(crate) async unsafe extern "C" fn foo()
            ```
        "#]],
    );
}

#[test]
fn test_hover_function_show_types() {
    check(
        r#"fn foo$0(a: i32, b:i32) -> i32 { 0 }"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn foo(a: i32, b: i32) -> i32
            ```
        "#]],
    );
}

#[test]
fn test_hover_function_associated_type_params() {
    check(
        r#"
trait Foo { type Bar; }
impl Foo for i32 { type Bar = i64; }
fn foo(arg: <i32 as Foo>::Bar) {}
fn main() { foo$0; }
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn foo(arg: <i32 as Foo>::Bar)
            ```
        "#]],
    );

    check(
        r#"
trait Foo<T> { type Bar<U>; }
impl Foo<i64> for i32 { type Bar<U> = i32; }
fn foo(arg: <<i32 as Foo<i64>>::Bar<i8> as Foo<i64>>::Bar<i8>) {}
fn main() { foo$0; }
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn foo(arg: <<i32 as Foo<i64>>::Bar<i8> as Foo<i64>>::Bar<i8>)
            ```
        "#]],
    );
}

#[test]
fn test_hover_function_pointer_show_identifiers() {
    check(
        r#"type foo$0 = fn(a: i32, b: i32) -> i32;"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            type foo = fn(a: i32, b: i32) -> i32
            ```

            ---

            size = 8, align = 8, niches = 1, no Drop
        "#]],
    );
}

#[test]
fn test_hover_function_pointer_no_identifier() {
    check(
        r#"type foo$0 = fn(i32, _: i32) -> i32;"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            type foo = fn(i32, i32) -> i32
            ```

            ---

            size = 8, align = 8, niches = 1, no Drop
        "#]],
    );
}

#[test]
fn test_hover_trait_show_qualifiers() {
    check_actions(
        r"unsafe trait foo$0() {}",
        expect![[r#"
            [
                Implementation(
                    FilePositionWrapper {
                        file_id: FileId(
                            0,
                        ),
                        offset: 13,
                    },
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_extern_crate() {
    check(
        r#"
//- /main.rs crate:main deps:std
//! Crate docs

/// Decl docs!
extern crate st$0d;
//- /std/lib.rs crate:std
//! Standard library for this test
//!
//! Printed?
//! abc123
"#,
        expect![[r#"
            *std*

            ```rust
            main
            ```

            ```rust
            extern crate std
            ```

            ---

            Decl docs!

            Standard library for this test

            Printed?
            abc123
        "#]],
    );
    check(
        r#"
//- /main.rs crate:main deps:std
//! Crate docs

/// Decl docs!
extern crate std as ab$0c;
//- /std/lib.rs crate:std
//! Standard library for this test
//!
//! Printed?
//! abc123
"#,
        expect![[r#"
            *abc*

            ```rust
            main
            ```

            ```rust
            extern crate std as abc
            ```

            ---

            Decl docs!

            Standard library for this test

            Printed?
            abc123
        "#]],
    );
}

#[test]
fn test_hover_mod_with_same_name_as_function() {
    check(
        r#"
use self::m$0y::Bar;
mod my { pub struct Bar; }

fn my() {}
"#,
        expect![[r#"
            *my*

            ```rust
            ra_test_fixture
            ```

            ```rust
            mod my
            ```
        "#]],
    );
}

#[test]
fn test_hover_struct_doc_comment() {
    check(
        r#"
/// This is an example
/// multiline doc
///
/// # Example
///
/// ```
/// let five = 5;
///
/// assert_eq!(6, my_crate::add_one(5));
/// ```
struct Bar;

fn foo() { let bar = Ba$0r; }
"#,
        expect![[r#"
            *Bar*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Bar
            ```

            ---

            This is an example
            multiline doc

            # Example

            ```
            let five = 5;

            assert_eq!(6, my_crate::add_one(5));
            ```
        "#]],
    );
}

#[test]
fn test_hover_struct_doc_attr() {
    check(
        r#"
#[doc = "bar docs"]
struct Bar;

fn foo() { let bar = Ba$0r; }
"#,
        expect![[r#"
            *Bar*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Bar
            ```

            ---

            bar docs
        "#]],
    );
}

#[test]
fn test_hover_struct_doc_attr_multiple_and_mixed() {
    check(
        r#"
/// bar docs 0
#[doc = "bar docs 1"]
#[doc = "bar docs 2"]
struct Bar;

fn foo() { let bar = Ba$0r; }
"#,
        expect![[r#"
            *Bar*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Bar
            ```

            ---

            bar docs 0
            bar docs 1
            bar docs 2
        "#]],
    );
}

#[test]
fn test_hover_external_url() {
    check(
        r#"
pub struct Foo;
/// [external](https://www.google.com)
pub struct B$0ar
"#,
        expect![[r#"
            *Bar*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub struct Bar
            ```

            ---

            size = 0, align = 1, no Drop

            ---

            [external](https://www.google.com)
        "#]],
    );
}

// Check that we don't rewrite links which we can't identify
#[test]
fn test_hover_unknown_target() {
    check(
        r#"
pub struct Foo;
/// [baz](Baz)
pub struct B$0ar
"#,
        expect![[r#"
            *Bar*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub struct Bar
            ```

            ---

            size = 0, align = 1, no Drop

            ---

            [baz](Baz)
        "#]],
    );
}

#[test]
fn test_hover_no_links() {
    check_hover_no_links(
        r#"
/// Test cases:
/// case 1.  bare URL: https://www.example.com/
/// case 2.  inline URL with title: [example](https://www.example.com/)
/// case 3.  code reference: [`Result`]
/// case 4.  code reference but miss footnote: [`String`]
/// case 5.  autolink: <http://www.example.com/>
/// case 6.  email address: <test@example.com>
/// case 7.  reference: [example][example]
/// case 8.  collapsed link: [example][]
/// case 9.  shortcut link: [example]
/// case 10. inline without URL: [example]()
/// case 11. reference: [foo][foo]
/// case 12. reference: [foo][bar]
/// case 13. collapsed link: [foo][]
/// case 14. shortcut link: [foo]
/// case 15. inline without URL: [foo]()
/// case 16. just escaped text: \[foo]
/// case 17. inline link: [Foo](foo::Foo)
///
/// [`Result`]: ../../std/result/enum.Result.html
/// [^example]: https://www.example.com/
pub fn fo$0o() {}
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub fn foo()
            ```

            ---

            Test cases:
            case 1.  bare URL: https://www.example.com/
            case 2.  inline URL with title: [example](https://www.example.com/)
            case 3.  code reference: `Result`
            case 4.  code reference but miss footnote: `String`
            case 5.  autolink: http://www.example.com/
            case 6.  email address: test@example.com
            case 7.  reference: example
            case 8.  collapsed link: example
            case 9.  shortcut link: example
            case 10. inline without URL: example
            case 11. reference: foo
            case 12. reference: foo
            case 13. collapsed link: foo
            case 14. shortcut link: foo
            case 15. inline without URL: foo
            case 16. just escaped text: \[foo\]
            case 17. inline link: Foo

            [^example]: https://www.example.com/
        "#]],
    );
}

#[test]
fn test_hover_layout_of_variant() {
    check(
        r#"enum Foo {
            Va$0riant1(u8, u16),
            Variant2(i32, u8, i64),
        }"#,
        expect![[r#"
            *Variant1*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            Variant1(u8, u16)
            ```

            ---

            size = 4, align = 2, no Drop
        "#]],
    );
}

#[test]
fn test_hover_layout_of_variant_generic() {
    check(
        r#"enum Option<T> {
    Some(T),
    None$0
}"#,
        expect![[r#"
            *None*

            ```rust
            ra_test_fixture::Option
            ```

            ```rust
            None
            ```

            ---

            no Drop
        "#]],
    );
}

#[test]
fn test_hover_layout_generic_unused() {
    check(
        r#"
//- minicore: phantom_data
struct S$0<T>(core::marker::PhantomData<T>);
"#,
        expect![[r#"
            *S*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct S<T>(PhantomData<T>)
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn test_hover_layout_of_enum() {
    check(
        r#"enum $0Foo {
            Variant1(u8, u16),
            Variant2(i32, u8, i64),
        }"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            enum Foo {
                Variant1( /* … */ ),
                Variant2( /* … */ ),
            }
            ```

            ---

            size = 16 (0x10), align = 8, niches = 254, no Drop
        "#]],
    );
}

#[test]
fn test_hover_no_memory_layout() {
    check_hover_no_memory_layout(
        r#"struct Foo { fiel$0d_a: u8, field_b: i32, field_c: i16 }"#,
        expect![[r#"
            *field_a*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            field_a: u8
            ```

            ---

            no Drop
        "#]],
    );

    check_hover_no_memory_layout(
        r#"
//- minicore: copy
fn main() {
    let x = 2;
    let y = $0|z| x + z;
}
"#,
        expect![[r#"
            *|*
            ```rust
            impl Fn(i32) -> i32
            ```

            ## Captures
            * `x` by immutable borrow
        "#]],
    );
}

#[test]
fn test_hover_macro_generated_struct_fn_doc_comment() {
    cov_mark::check!(hover_macro_generated_struct_fn_doc_comment);

    check(
        r#"
macro_rules! bar {
    () => {
        struct Bar;
        impl Bar {
            /// Do the foo
            fn foo(&self) {}
        }
    }
}

bar!();

fn foo() { let bar = Bar; bar.fo$0o(); }
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::Bar
            ```

            ```rust
            fn foo(&self)
            ```

            ---

            Do the foo
        "#]],
    );
}

#[test]
fn test_hover_macro_generated_struct_fn_doc_attr() {
    cov_mark::check!(hover_macro_generated_struct_fn_doc_attr);

    check(
        r#"
macro_rules! bar {
    () => {
        struct Bar;
        impl Bar {
            #[doc = "Do the foo"]
            fn foo(&self) {}
        }
    }
}

bar!();

fn foo() { let bar = Bar; bar.fo$0o(); }
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::Bar
            ```

            ```rust
            fn foo(&self)
            ```

            ---

            Do the foo
        "#]],
    );
}

#[test]
fn test_hover_variadic_function() {
    check(
        r#"
extern "C" {
    pub fn foo(bar: i32, ...) -> i32;
}

fn main() { let foo_test = unsafe { fo$0o(1, 2, 3); } }
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::<extern>
            ```

            ```rust
            pub unsafe fn foo(bar: i32, ...) -> i32
            ```
        "#]],
    );
}

#[test]
fn test_hover_trait_has_impl_action() {
    check_actions(
        r#"trait foo$0() {}"#,
        expect![[r#"
            [
                Implementation(
                    FilePositionWrapper {
                        file_id: FileId(
                            0,
                        ),
                        offset: 6,
                    },
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_struct_has_impl_action() {
    check_actions(
        r"struct foo$0() {}",
        expect![[r#"
            [
                Implementation(
                    FilePositionWrapper {
                        file_id: FileId(
                            0,
                        ),
                        offset: 7,
                    },
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_union_has_impl_action() {
    check_actions(
        r#"union foo$0() {}"#,
        expect![[r#"
            [
                Implementation(
                    FilePositionWrapper {
                        file_id: FileId(
                            0,
                        ),
                        offset: 6,
                    },
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_enum_has_impl_action() {
    check_actions(
        r"enum foo$0() { A, B }",
        expect![[r#"
            [
                Implementation(
                    FilePositionWrapper {
                        file_id: FileId(
                            0,
                        ),
                        offset: 5,
                    },
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_self_has_impl_action() {
    check_actions(
        r#"struct foo where Self$0:;"#,
        expect![[r#"
            [
                Implementation(
                    FilePositionWrapper {
                        file_id: FileId(
                            0,
                        ),
                        offset: 7,
                    },
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_test_has_action() {
    check_actions(
        r#"
#[test]
fn foo_$0test() {}
"#,
        expect![[r#"
            [
                Reference(
                    FilePositionWrapper {
                        file_id: FileId(
                            0,
                        ),
                        offset: 11,
                    },
                ),
                Runnable(
                    Runnable {
                        use_name_in_title: false,
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 0..24,
                            focus_range: 11..19,
                            name: "foo_test",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "foo_test",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg: None,
                        update_test: UpdateTest {
                            expect_test: false,
                            insta: false,
                            snapbox: false,
                        },
                    },
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_test_mod_has_action() {
    check_actions(
        r#"
mod tests$0 {
    #[test]
    fn foo_test() {}
}
"#,
        expect![[r#"
            [
                Runnable(
                    Runnable {
                        use_name_in_title: false,
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 0..46,
                            focus_range: 4..9,
                            name: "tests",
                            kind: Module,
                            description: "mod tests",
                        },
                        kind: TestMod {
                            path: "tests",
                        },
                        cfg: None,
                        update_test: UpdateTest {
                            expect_test: false,
                            insta: false,
                            snapbox: false,
                        },
                    },
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_struct_has_goto_type_action() {
    check_actions(
        r#"
struct S{ f1: u32 }

fn main() { let s$0t = S{ f1:0 }; }
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..19,
                                focus_range: 7..8,
                                name: "S",
                                kind: Struct,
                                description: "struct S",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_generic_struct_has_goto_type_actions() {
    check_actions(
        r#"
struct Arg(u32);
struct S<T>{ f1: T }

fn main() { let s$0t = S{ f1:Arg(0) }; }
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Arg",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..16,
                                focus_range: 7..10,
                                name: "Arg",
                                kind: Struct,
                                description: "struct Arg(u32)",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 17..37,
                                focus_range: 24..25,
                                name: "S",
                                kind: Struct,
                                description: "struct S<T>",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_generic_excludes_sized_go_to_action() {
    check_actions(
        r#"
//- minicore: sized
struct S<T$0>(T);
    "#,
        expect![[r#"
            []
        "#]],
    );
}

#[test]
fn test_hover_generic_struct_has_flattened_goto_type_actions() {
    check_actions(
        r#"
struct Arg(u32);
struct S<T>{ f1: T }

fn main() { let s$0t = S{ f1: S{ f1: Arg(0) } }; }
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Arg",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..16,
                                focus_range: 7..10,
                                name: "Arg",
                                kind: Struct,
                                description: "struct Arg(u32)",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 17..37,
                                focus_range: 24..25,
                                name: "S",
                                kind: Struct,
                                description: "struct S<T>",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_tuple_has_goto_type_actions() {
    check_actions(
        r#"
struct A(u32);
struct B(u32);
mod M {
    pub struct C(u32);
}

fn main() { let s$0t = (A(1), B(2), M::C(3) ); }
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::A",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..14,
                                focus_range: 7..8,
                                name: "A",
                                kind: Struct,
                                description: "struct A(u32)",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::B",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 15..29,
                                focus_range: 22..23,
                                name: "B",
                                kind: Struct,
                                description: "struct B(u32)",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::M::C",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 42..60,
                                focus_range: 53..54,
                                name: "C",
                                kind: Struct,
                                container_name: "M",
                                description: "pub struct C(u32)",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_return_impl_trait_has_goto_type_action() {
    check_actions(
        r#"
trait Foo {}
fn foo() -> impl Foo {}

fn main() { let s$0t = foo(); }
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..12,
                                focus_range: 6..9,
                                name: "Foo",
                                kind: Trait,
                                description: "trait Foo",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_generic_return_impl_trait_has_goto_type_action() {
    check_actions(
        r#"
trait Foo<T> {}
struct S;
fn foo() -> impl Foo<S> {}

fn main() { let s$0t = foo(); }
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..15,
                                focus_range: 6..9,
                                name: "Foo",
                                kind: Trait,
                                description: "trait Foo<T>",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 16..25,
                                focus_range: 23..24,
                                name: "S",
                                kind: Struct,
                                description: "struct S",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_return_impl_traits_has_goto_type_action() {
    check_actions(
        r#"
trait Foo {}
trait Bar {}
fn foo() -> impl Foo + Bar {}

fn main() { let s$0t = foo(); }
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Bar",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 13..25,
                                focus_range: 19..22,
                                name: "Bar",
                                kind: Trait,
                                description: "trait Bar",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..12,
                                focus_range: 6..9,
                                name: "Foo",
                                kind: Trait,
                                description: "trait Foo",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_generic_return_impl_traits_has_goto_type_action() {
    check_actions(
        r#"
trait Foo<T> {}
trait Bar<T> {}
struct S1 {}
struct S2 {}

fn foo() -> impl Foo<S1> + Bar<S2> {}

fn main() { let s$0t = foo(); }
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Bar",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 16..31,
                                focus_range: 22..25,
                                name: "Bar",
                                kind: Trait,
                                description: "trait Bar<T>",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..15,
                                focus_range: 6..9,
                                name: "Foo",
                                kind: Trait,
                                description: "trait Foo<T>",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S1",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 32..44,
                                focus_range: 39..41,
                                name: "S1",
                                kind: Struct,
                                description: "struct S1",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S2",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 45..57,
                                focus_range: 52..54,
                                name: "S2",
                                kind: Struct,
                                description: "struct S2",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_arg_impl_trait_has_goto_type_action() {
    check_actions(
        r#"
trait Foo {}
fn foo(ar$0g: &impl Foo) {}
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..12,
                                focus_range: 6..9,
                                name: "Foo",
                                kind: Trait,
                                description: "trait Foo",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_arg_impl_traits_has_goto_type_action() {
    check_actions(
        r#"
trait Foo {}
trait Bar<T> {}
struct S{}

fn foo(ar$0g: &impl Foo + Bar<S>) {}
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Bar",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 13..28,
                                focus_range: 19..22,
                                name: "Bar",
                                kind: Trait,
                                description: "trait Bar<T>",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..12,
                                focus_range: 6..9,
                                name: "Foo",
                                kind: Trait,
                                description: "trait Foo",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 29..39,
                                focus_range: 36..37,
                                name: "S",
                                kind: Struct,
                                description: "struct S",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_async_block_impl_trait_has_goto_type_action() {
    check_actions(
        r#"
//- /main.rs crate:main deps:core
// we don't use minicore here so that this test doesn't randomly fail
// when someone edits minicore
struct S;
fn foo() {
    let fo$0o = async { S };
}
//- /core.rs crate:core
pub mod future {
    #[lang = "future_trait"]
    pub trait Future {}
}
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "core::future::Future",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 4294967295..4294967295,
                                focus_range: 4294967295..4294967295,
                                name: "Future",
                                kind: Trait,
                                container_name: "future",
                                description: "pub trait Future",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "main::S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..110,
                                focus_range: 108..109,
                                name: "S",
                                kind: Struct,
                                description: "struct S",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_arg_generic_impl_trait_has_goto_type_action() {
    check_actions(
        r#"
trait Foo<T> {}
struct S {}
fn foo(ar$0g: &impl Foo<S>) {}
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..15,
                                focus_range: 6..9,
                                name: "Foo",
                                kind: Trait,
                                description: "trait Foo<T>",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 16..27,
                                focus_range: 23..24,
                                name: "S",
                                kind: Struct,
                                description: "struct S",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_dyn_return_has_goto_type_action() {
    check_actions(
        r#"
trait Foo<T> {}
struct S;
impl Foo<S> for S {}

struct B<T>{}
fn foo() -> B<dyn Foo<S>> {}

fn main() { let s$0t = foo(); }
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::B",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 48..61,
                                focus_range: 55..56,
                                name: "B",
                                kind: Struct,
                                description: "struct B<T>",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..15,
                                focus_range: 6..9,
                                name: "Foo",
                                kind: Trait,
                                description: "trait Foo<T>",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 16..25,
                                focus_range: 23..24,
                                name: "S",
                                kind: Struct,
                                description: "struct S",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_dyn_arg_has_goto_type_action() {
    check_actions(
        r#"
trait Foo {}
fn foo(ar$0g: &dyn Foo) {}
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..12,
                                focus_range: 6..9,
                                name: "Foo",
                                kind: Trait,
                                description: "trait Foo",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_generic_dyn_arg_has_goto_type_action() {
    check_actions(
        r#"
trait Foo<T> {}
struct S {}
fn foo(ar$0g: &dyn Foo<S>) {}
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..15,
                                focus_range: 6..9,
                                name: "Foo",
                                kind: Trait,
                                description: "trait Foo<T>",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 16..27,
                                focus_range: 23..24,
                                name: "S",
                                kind: Struct,
                                description: "struct S",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_goto_type_action_links_order() {
    check_actions(
        r#"
trait ImplTrait<T> {}
trait DynTrait<T> {}
struct B<T> {}
struct S {}

fn foo(a$0rg: &impl ImplTrait<B<dyn DynTrait<B<S>>>>) {}
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::B",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 43..57,
                                focus_range: 50..51,
                                name: "B",
                                kind: Struct,
                                description: "struct B<T>",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::DynTrait",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 22..42,
                                focus_range: 28..36,
                                name: "DynTrait",
                                kind: Trait,
                                description: "trait DynTrait<T>",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::ImplTrait",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..21,
                                focus_range: 6..15,
                                name: "ImplTrait",
                                kind: Trait,
                                description: "trait ImplTrait<T>",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 58..69,
                                focus_range: 65..66,
                                name: "S",
                                kind: Struct,
                                description: "struct S",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_associated_type_has_goto_type_action() {
    check_actions(
        r#"
trait Foo {
    type Item;
    fn get(self) -> Self::Item {}
}

struct Bar{}
struct S{}

impl Foo for S { type Item = Bar; }

fn test() -> impl Foo { S {} }

fn main() { let s$0t = test().get(); }
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..62,
                                focus_range: 6..9,
                                name: "Foo",
                                kind: Trait,
                                description: "trait Foo",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_const_param_has_goto_type_action() {
    check_actions(
        r#"
struct Bar;
struct Foo<const BAR: Bar>;

impl<const BAR: Bar> Foo<BAR$0> {}
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Bar",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..11,
                                focus_range: 7..10,
                                name: "Bar",
                                kind: Struct,
                                description: "struct Bar",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_type_param_has_goto_type_action() {
    check_actions(
        r#"
trait Foo {}

fn foo<T: Foo>(t: T$0){}
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..12,
                                focus_range: 6..9,
                                name: "Foo",
                                kind: Trait,
                                description: "trait Foo",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn test_hover_self_has_go_to_type() {
    check_actions(
        r#"
struct Foo;
impl Foo {
    fn foo(&self$0) {}
}
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..11,
                                focus_range: 7..10,
                                name: "Foo",
                                kind: Struct,
                                description: "struct Foo",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn hover_displays_normalized_crate_names() {
    check(
        r#"
//- /lib.rs crate:name-with-dashes
pub mod wrapper {
    pub struct Thing { x: u32 }

    impl Thing {
        pub fn new() -> Thing { Thing { x: 0 } }
    }
}

//- /main.rs crate:main deps:name-with-dashes
fn main() { let foo_test = name_with_dashes::wrapper::Thing::new$0(); }
"#,
        expect![[r#"
            *new*

            ```rust
            name_with_dashes::wrapper::Thing
            ```

            ```rust
            pub fn new() -> Thing
            ```
        "#]],
    )
}

#[test]
fn hover_field_pat_shorthand_ref_match_ergonomics() {
    check(
        r#"
struct S {
    f: i32,
}

fn main() {
    let s = S { f: 0 };
    let S { f$0 } = &s;
}
"#,
        expect![[r#"
            *f*

            ```rust
            let f: &i32
            ```

            ---

            size = 8, align = 8, niches = 1, no Drop

            ---

            ```rust
            ra_test_fixture::S
            ```

            ```rust
            f: i32
            ```

            ---

            size = 4, align = 4, offset = 0, no Drop
        "#]],
    );
}

#[test]
fn const_generic_order() {
    check(
        r#"
struct Foo;
struct S$0T<const C: usize = 1, T = Foo>(T);
"#,
        expect![[r#"
            *ST*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct ST<const C: usize = {const}, T = Foo>(T)
            ```

            ---

            size = 0, align = 1, type param may need Drop
        "#]],
    );
}

#[test]
fn const_generic_default_value() {
    check(
        r#"
struct Foo;
struct S$0T<const C: usize = {40 + 2}, T = Foo>(T);
"#,
        expect![[r#"
            *ST*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct ST<const C: usize = {const}, T = Foo>(T)
            ```

            ---

            size = 0, align = 1, type param may need Drop
        "#]],
    );
}

#[test]
fn const_generic_default_value_2() {
    check(
        r#"
struct Foo;
const VAL = 1;
struct S$0T<const C: usize = VAL, T = Foo>(T);
"#,
        expect![[r#"
            *ST*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct ST<const C: usize = {const}, T = Foo>(T)
            ```

            ---

            size = 0, align = 1, type param may need Drop
        "#]],
    );
}

#[test]
fn const_generic_positive_i8_literal() {
    check(
        r#"
struct Const<const N: i8>;

fn main() {
    let v$0alue = Const::<1>;
}
"#,
        expect![[r#"
            *value*

            ```rust
            let value: Const<1>
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn const_generic_zero_i8_literal() {
    check(
        r#"
struct Const<const N: i8>;

fn main() {
    let v$0alue = Const::<0>;
}
"#,
        expect![[r#"
            *value*

            ```rust
            let value: Const<0>
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn const_generic_negative_i8_literal() {
    check(
        r#"
struct Const<const N: i8>;

fn main() {
    let v$0alue = Const::<-1>;
}
"#,
        expect![[r#"
            *value*

            ```rust
            let value: Const<_>
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn const_generic_bool_literal() {
    check(
        r#"
struct Const<const F: bool>;

fn main() {
    let v$0alue = Const::<true>;
}
"#,
        expect![[r#"
            *value*

            ```rust
            let value: Const<true>
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn const_generic_char_literal() {
    check(
        r#"
struct Const<const C: char>;

fn main() {
    let v$0alue = Const::<'🦀'>;
}
"#,
        expect![[r#"
            *value*

            ```rust
            let value: Const<'🦀'>
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn hover_self_param_shows_type() {
    check(
        r#"
struct Foo {}
impl Foo {
    fn bar(&sel$0f) {}
}
"#,
        expect![[r#"
            *self*

            ```rust
            self: &Foo
            ```

            ---

            size = 8, align = 8, niches = 1, no Drop
        "#]],
    );
}

#[test]
fn hover_self_param_shows_type_for_arbitrary_self_type() {
    check(
        r#"
struct Arc<T>(T);
struct Foo {}
impl Foo {
    fn bar(sel$0f: Arc<Foo>) {}
}
"#,
        expect![[r#"
            *self*

            ```rust
            self: Arc<Foo>
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn hover_doc_outer_inner() {
    check(
        r#"
/// Be quick;
mod Foo$0 {
    //! time is mana

    /// This comment belongs to the function
    fn foo() {}
}
"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            mod Foo
            ```

            ---

            Be quick;
            time is mana
        "#]],
    );
}

#[test]
fn hover_doc_outer_inner_attribute() {
    check(
        r#"
#[doc = "Be quick;"]
mod Foo$0 {
    #![doc = "time is mana"]

    #[doc = "This comment belongs to the function"]
    fn foo() {}
}
"#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            mod Foo
            ```

            ---

            Be quick;
            time is mana
        "#]],
    );
}

#[test]
fn hover_doc_block_style_indent_end() {
    check(
        r#"
/**
    foo
    ```rust
    let x = 3;
    ```
*/
fn foo$0() {}
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn foo()
            ```

            ---

            foo

            ```rust
            let x = 3;
            ```
        "#]],
    );
}

#[test]
fn hover_comments_dont_highlight_parent() {
    cov_mark::check!(no_highlight_on_comment_hover);
    check_hover_no_result(
        r#"
fn no_hover() {
    // no$0hover
}
"#,
    );
}

#[test]
fn hover_label() {
    check(
        r#"
fn foo() {
    'label$0: loop {}
}
"#,
        expect![[r#"
            *'label*

            ```rust
            'label
            ```
        "#]],
    );
}

#[test]
fn hover_lifetime() {
    check(
        r#"fn foo<'lifetime>(_: &'lifetime$0 ()) {}"#,
        expect![[r#"
            *'lifetime*

            ```rust
            ra_test_fixture::foo
            ```

            ```rust
            'lifetime
            ```
        "#]],
    );
    check(
        r#"fn foo(_: &'static$0 ()) {}"#,
        expect![[r#"
            *'static*

            ```rust
            'static
            ```
        "#]],
    );
}

#[test]
fn hover_type_param() {
    check(
        r#"
//- minicore: sized
struct Foo<T>(T);
trait TraitA {}
trait TraitB {}
impl<T: TraitA + TraitB> Foo<T$0> where T: Sized {}
"#,
        expect![[r#"
            *T*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            T: TraitA + TraitB
            ```
        "#]],
    );
    check(
        r#"
//- minicore: sized
struct Foo<T>(T);
impl<T> Foo<T$0> {}
"#,
        expect![[r#"
            *T*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            T
            ```
        "#]],
    );
    check(
        r#"
//- minicore: sized
struct Foo<T>(T);
impl<T: 'static> Foo<T$0> {}
"#,
        expect![[r#"
            *T*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            T: 'static
            ```
        "#]],
    );
}

#[test]
fn hover_type_param_sized_bounds() {
    // implicit `: Sized` bound
    check(
        r#"
//- minicore: sized
trait Trait {}
struct Foo<T>(T);
impl<T$0: Trait> Foo<T> {}
"#,
        expect![[r#"
            *T*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            T: Trait
            ```
        "#]],
    );
    check(
        r#"
//- minicore: sized
trait Trait {}
struct Foo<T>(T);
impl<T$0: Trait + ?Sized> Foo<T> {}
"#,
        expect![[r#"
            *T*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            T: Trait + ?Sized
            ```
        "#]],
    );
}

mod type_param_sized_bounds {
    use super::*;

    #[test]
    fn single_implicit() {
        check(
            r#"
//- minicore: sized
fn foo<T$0>() {}
"#,
            expect![[r#"
                *T*

                ```rust
                ra_test_fixture::foo
                ```

                ```rust
                T
                ```

                ---

                invariant
            "#]],
        );
    }

    #[test]
    fn single_explicit() {
        check(
            r#"
//- minicore: sized
fn foo<T$0: Sized>() {}
"#,
            expect![[r#"
                *T*

                ```rust
                ra_test_fixture::foo
                ```

                ```rust
                T
                ```

                ---

                invariant
            "#]],
        );
    }

    #[test]
    fn single_relaxed() {
        check(
            r#"
//- minicore: sized
fn foo<T$0: ?Sized>() {}
"#,
            expect![[r#"
                *T*

                ```rust
                ra_test_fixture::foo
                ```

                ```rust
                T: ?Sized
                ```

                ---

                invariant
            "#]],
        );
    }

    #[test]
    fn multiple_implicit() {
        check(
            r#"
//- minicore: sized
trait Trait {}
fn foo<T$0: Trait>() {}
"#,
            expect![[r#"
                *T*

                ```rust
                ra_test_fixture::foo
                ```

                ```rust
                T: Trait
                ```

                ---

                invariant
            "#]],
        );
    }

    #[test]
    fn multiple_explicit() {
        check(
            r#"
//- minicore: sized
trait Trait {}
fn foo<T$0: Trait + Sized>() {}
"#,
            expect![[r#"
                *T*

                ```rust
                ra_test_fixture::foo
                ```

                ```rust
                T: Trait
                ```

                ---

                invariant
            "#]],
        );
    }

    #[test]
    fn multiple_relaxed() {
        check(
            r#"
//- minicore: sized
trait Trait {}
fn foo<T$0: Trait + ?Sized>() {}
"#,
            expect![[r#"
                *T*

                ```rust
                ra_test_fixture::foo
                ```

                ```rust
                T: Trait + ?Sized
                ```

                ---

                invariant
            "#]],
        );
    }

    #[test]
    fn mixed() {
        check(
            r#"
//- minicore: sized
fn foo<T$0: ?Sized + Sized + Sized>() {}
"#,
            expect![[r#"
                *T*

                ```rust
                ra_test_fixture::foo
                ```

                ```rust
                T
                ```

                ---

                invariant
            "#]],
        );
    }

    #[test]
    fn mixed2() {
        check(
            r#"
//- minicore: sized
trait Trait {}
fn foo<T$0: Sized + ?Sized + Sized + Trait>() {}
"#,
            expect![[r#"
                *T*

                ```rust
                ra_test_fixture::foo
                ```

                ```rust
                T: Trait
                ```

                ---

                invariant
            "#]],
        );
    }
}

#[test]
fn hover_const_generic_type_alias() {
    check(
        r#"
struct Foo<const LEN: usize>;
type Fo$0o2 = Foo<2>;
"#,
        expect![[r#"
            *Foo2*

            ```rust
            ra_test_fixture
            ```

            ```rust
            type Foo2 = Foo<<expr>>
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn hover_const_param() {
    check(
        r#"
struct Foo<const LEN: usize>;
impl<const LEN: usize> Foo<LEN$0> {}
"#,
        expect![[r#"
            *LEN*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            const LEN: usize
            ```
        "#]],
    );
}

#[test]
fn hover_const_eval_discriminant() {
    // Don't show hex for <10
    check(
        r#"
#[repr(u8)]
enum E {
    /// This is a doc
    A$0 = 1 << 3,
}
"#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture::E
            ```

            ```rust
            A = 8
            ```

            ---

            size = 1, align = 1, no Drop

            ---

            This is a doc
        "#]],
    );
    // Show hex for >10
    check(
        r#"
#[repr(u8)]
enum E {
    /// This is a doc
    A$0 = (1 << 3) + (1 << 2),
}
"#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture::E
            ```

            ```rust
            A = 12 (0xC)
            ```

            ---

            size = 1, align = 1, no Drop

            ---

            This is a doc
        "#]],
    );
    // enums in const eval
    check(
        r#"
#[repr(u8)]
enum E {
    A = 1,
    /// This is a doc
    B$0 = E::A as u8 + 1,
}
"#,
        expect![[r#"
            *B*

            ```rust
            ra_test_fixture::E
            ```

            ```rust
            B = 2
            ```

            ---

            size = 1, align = 1, no Drop

            ---

            This is a doc
        "#]],
    );
    // unspecified variant should increment by one
    check(
        r#"
#[repr(u8)]
enum E {
    A = 4,
    /// This is a doc
    B$0,
}
"#,
        expect![[r#"
            *B*

            ```rust
            ra_test_fixture::E
            ```

            ```rust
            B = 5
            ```

            ---

            size = 1, align = 1, no Drop

            ---

            This is a doc
        "#]],
    );
}

#[test]
fn hover_const_eval() {
    check(
        r#"
trait T {
    const B: bool = false;
}
impl T for <()> {
    /// true
    const B: bool = true;
}
fn main() {
    <()>::B$0;
}
"#,
        expect![[r#"
            *B*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const B: bool = true
            ```

            ---

            true
        "#]],
    );

    check(
        r#"
struct A {
    i: i32
};

trait T {
    const AA: A = A {
        i: 1
    };
}
impl T for i32 {
    const AA: A = A {
        i: 2 + 3
    }
}
fn main() {
    <i32>::AA$0;
}
"#,
        expect![[r#"
            *AA*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const AA: A = A { i: 5 }
            ```
        "#]],
    );

    check(
        r#"
trait T {
    /// false
    const B: bool = false;
}
impl T for () {
    /// true
    const B: bool = true;
}
fn main() {
    T::B$0;
}
"#,
        expect![[r#"
            *B*

            ```rust
            ra_test_fixture::T
            ```

            ```rust
            const B: bool = false
            ```

            ---

            false
        "#]],
    );

    check(
        r#"
trait T {
    /// false
    const B: bool = false;
}
impl T for () {
}
fn main() {
    <()>::B$0;
}
"#,
        expect![[r#"
            *B*

            ```rust
            ra_test_fixture::T
            ```

            ```rust
            const B: bool = false
            ```

            ---

            `Self` = `()`

            ---

            false
        "#]],
    );

    check(
        r#"
trait T {
    /// false
    const B: bool = false;
}
impl T for () {
    /// true
    const B: bool = true;
}
impl T for i32 {}
fn main() {
    <i32>::B$0;
}
"#,
        expect![[r#"
            *B*

            ```rust
            ra_test_fixture::T
            ```

            ```rust
            const B: bool = false
            ```

            ---

            `Self` = `i32`

            ---

            false
        "#]],
    );

    // show hex for <10
    check(
        r#"
/// This is a doc
const FOO$0: usize = 1 << 3;
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: usize = 8
            ```

            ---

            This is a doc
        "#]],
    );
    check(
        r#"
/// This is a doc
const FOO$0: usize = (1 << 3) + (1 << 2);
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: usize = 12 (0xC)
            ```

            ---

            This is a doc
        "#]],
    );
    // show original body when const eval fails
    check(
        r#"
/// This is a doc
const FOO$0: usize = 2 - 3;
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: usize = 2 - 3
            ```

            ---

            This is a doc
        "#]],
    );
    // don't show hex for negatives
    check(
        r#"
/// This is a doc
const FOO$0: i32 = 2 - 3;
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: i32 = -1 (0xFFFFFFFF)
            ```

            ---

            This is a doc
        "#]],
    );
    check(
        r#"
/// This is a doc
const FOO$0: &str = "bar";
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: &str = "bar"
            ```

            ---

            This is a doc
        "#]],
    );
    // show char literal
    check(
        r#"
/// This is a doc
const FOO$0: char = 'a';
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: char = 'a'
            ```

            ---

            This is a doc
        "#]],
    );
    // show escaped char literal
    check(
        r#"
/// This is a doc
const FOO$0: char = '\x61';
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: char = 'a'
            ```

            ---

            This is a doc
        "#]],
    );
    // show byte literal
    check(
        r#"
/// This is a doc
const FOO$0: u8 = b'a';
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: u8 = 97 (0x61)
            ```

            ---

            This is a doc
        "#]],
    );
    // show escaped byte literal
    check(
        r#"
/// This is a doc
const FOO$0: u8 = b'\x61';
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: u8 = 97 (0x61)
            ```

            ---

            This is a doc
        "#]],
    );
    // show float literal
    check(
        r#"
    /// This is a doc
    const FOO$0: f64 = 1.0234;
    "#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: f64 = 1.0234
            ```

            ---

            This is a doc
        "#]],
    );
    //show float typecasted from int
    check(
        r#"
/// This is a doc
const FOO$0: f32 = 1f32;
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: f32 = 1.0
            ```

            ---

            This is a doc
        "#]],
    );
    // Don't show `<ref-not-supported>` in const hover
    check(
        r#"
/// This is a doc
const FOO$0: &i32 = &2;
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: &i32 = &2
            ```

            ---

            This is a doc
        "#]],
    );
    //show f64 typecasted from float
    check(
        r#"
/// This is a doc
const FOO$0: f64 = 1.0f64;
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: f64 = 1.0
            ```

            ---

            This is a doc
        "#]],
    );
}

#[test]
fn hover_const_eval_floating_point() {
    check(
        r#"
extern "rust-intrinsic" {
    pub fn expf64(x: f64) -> f64;
}

const FOO$0: f64 = expf64(1.2);
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: f64 = 3.3201169227365472
            ```
        "#]],
    );
    // check `f32` isn't double rounded via `f64`
    check(
        r#"
/// This is a doc
const FOO$0: f32 = 1.9999999403953552_f32;
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: f32 = 1.9999999
            ```

            ---

            This is a doc
        "#]],
    );
    // Check `f16` and `f128`
    check(
        r#"
/// This is a doc
const FOO$0: f16 = -1.0f16;
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: f16 = -1.0
            ```

            ---

            This is a doc
        "#]],
    );
    check(
        r#"
/// This is a doc
const FOO$0: f128 = -1.0f128;
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: f128 = -1.0
            ```

            ---

            This is a doc
        "#]],
    );
}

#[test]
fn hover_const_eval_enum() {
    check(
        r#"
enum Enum {
    V1,
    V2,
}

const VX: Enum = Enum::V1;

const FOO$0: Enum = VX;
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: Enum = V1
            ```
        "#]],
    );
    check(
        r#"
//- minicore: option
const FOO$0: Option<i32> = Some(2);
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: Option<i32> = Some(2)
            ```
        "#]],
    );
    check(
        r#"
//- minicore: option
const FOO$0: Option<&i32> = Some(2).as_ref();
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: Option<&i32> = Some(&2)
            ```
        "#]],
    );
}

#[test]
fn hover_const_eval_dyn_trait() {
    check(
        r#"
//- minicore: fmt, coerce_unsized, builtin_impls, dispatch_from_dyn
use core::fmt::Debug;

const FOO$0: &dyn Debug = &2i32;
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: &dyn Debug = &2
            ```
        "#]],
    );
}

#[test]
fn hover_const_eval_slice() {
    check(
        r#"
//- minicore: slice, index, coerce_unsized
const FOO$0: &[i32] = &[1, 2, 3 + 4];
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: &[i32] = &[1, 2, 7]
            ```
        "#]],
    );
    check(
        r#"
//- minicore: slice, index, coerce_unsized
const FOO$0: &[i32; 5] = &[12; 5];
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: &[i32; {const}] = &[12, 12, 12, 12, 12]
            ```
        "#]],
    );
    check(
        r#"
//- minicore: slice, index, coerce_unsized

const FOO$0: (&i32, &[i32], &i32) = {
    let a: &[i32] = &[1, 2, 3];
    (&a[0], a, &a[0])
}
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: (&i32, &[i32], &i32) = (&1, &[1, 2, 3], &1)
            ```
        "#]],
    );
    check(
        r#"
//- minicore: slice, index, coerce_unsized

struct Tree(&[Tree]);

const FOO$0: Tree = {
    let x = &[Tree(&[]), Tree(&[Tree(&[])])];
    Tree(&[Tree(x), Tree(x)])
}
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: Tree = Tree(&[Tree(&[Tree(&[]), Tree(&[Tree(&[])])]), Tree(&[Tree(&[]), Tree(&[Tree(&[])])])])
            ```
        "#]],
    );
    // FIXME: Show the data of unsized structs
    check(
        r#"
//- minicore: slice, index, coerce_unsized, transmute
#[repr(transparent)]
struct S<T: ?Sized>(T);
const FOO$0: &S<[u8]> = core::mem::transmute::<&[u8], _>(&[1, 2, 3]);
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: &S<[u8]> = &S
            ```
        "#]],
    );
}

#[test]
fn hover_const_eval_str() {
    check(
        r#"
const FOO$0: &str = "foo";
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: &str = "foo"
            ```
        "#]],
    );
    check(
        r#"
struct X {
    a: &'static str,
    b: &'static str,
}
const FOO$0: X = X {
    a: "axiom",
    b: "buy N large",
};
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: X = X { a: "axiom", b: "buy N large" }
            ```
        "#]],
    );
    check(
        r#"
const FOO$0: (&str, &str) = {
    let x = "foo";
    (x, x)
};
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: (&str, &str) = ("foo", "foo")
            ```
        "#]],
    );
}

#[test]
fn hover_const_eval_in_generic_trait() {
    // Doesn't compile, but we shouldn't crash.
    check(
        r#"
trait Trait<T> {
    const FOO: bool = false;
}
struct S<T>(T);
impl<T> Trait<T> for S<T> {
    const FOO: bool = true;
}

fn test() {
    S::FOO$0;
}
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture::S
            ```

            ```rust
            const FOO: bool = true
            ```
        "#]],
    );
}

#[test]
fn hover_const_pat() {
    check(
        r#"
/// This is a doc
const FOO: usize = 3;
fn foo() {
    match 5 {
        FOO$0 => (),
        _ => ()
    }
}
"#,
        expect![[r#"
            *FOO*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const FOO: usize = 3
            ```

            ---

            This is a doc
        "#]],
    );
    check(
        r#"
enum E {
    /// This is a doc
    A = 3,
}
fn foo(e: E) {
    match e {
        E::A$0 => (),
        _ => ()
    }
}
"#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture::E
            ```

            ```rust
            A = 3
            ```

            ---

            This is a doc
        "#]],
    );
}

#[test]
fn hover_const_value() {
    check(
        r#"
pub enum AA {
    BB,
}
const CONST: AA = AA::BB;
pub fn the_function() -> AA {
    CON$0ST
}
"#,
        expect![[r#"
            *CONST*

            ```rust
            ra_test_fixture
            ```

            ```rust
            const CONST: AA = BB
            ```
        "#]],
    );
}

#[test]
fn array_repeat_exp() {
    check(
        r#"
fn main() {
    let til$0e4 = [0_u32; (4 * 8 * 8) / 32];
}
        "#,
        expect![[r#"
            *tile4*

            ```rust
            let tile4: [u32; 8]
            ```

            ---

            size = 32 (0x20), align = 4, no Drop
        "#]],
    );
}

#[test]
fn hover_mod_def() {
    check(
        r#"
//- /main.rs
mod foo$0;
//- /foo.rs
//! For the horde!
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            mod foo
            ```

            ---

            For the horde!
        "#]],
    );
}

#[test]
fn hover_self_in_use() {
    check(
        r#"
//! This should not appear
mod foo {
    /// But this should appear
    pub mod bar {}
}
use foo::bar::{self$0};
"#,
        expect![[r#"
            *self*

            ```rust
            ra_test_fixture::foo
            ```

            ```rust
            pub mod bar
            ```

            ---

            But this should appear
        "#]],
    )
}

#[test]
fn hover_keyword() {
    check(
        r#"
//- /main.rs crate:main deps:std
fn f() { retur$0n; }
//- /libstd.rs crate:std
/// Docs for return_keyword
mod return_keyword {}
"#,
        expect![[r#"
                *return*

                ```rust
                return
                ```

                ---

                Docs for return_keyword
            "#]],
    );
}

#[test]
fn hover_keyword_doc() {
    check(
        r#"
//- /main.rs crate:main deps:std
fn foo() {
    let bar = mov$0e || {};
}
//- /libstd.rs crate:std
#[doc(keyword = "move")]
/// [closure]
/// [closures][closure]
/// [threads]
/// <https://doc.rust-lang.org/nightly/book/ch13-01-closures.html>
///
/// [closure]: ../book/ch13-01-closures.html
/// [threads]: ../book/ch16-01-threads.html#using-move-closures-with-threads
mod move_keyword {}
"#,
        expect![[r#"
            *move*

            ```rust
            move
            ```

            ---

            [closure](https://doc.rust-lang.org/stable/book/ch13-01-closures.html)
            [closures](https://doc.rust-lang.org/stable/book/ch13-01-closures.html)
            [threads](https://doc.rust-lang.org/stable/book/ch16-01-threads.html#using-move-closures-with-threads)
            <https://doc.rust-lang.org/nightly/book/ch13-01-closures.html>
        "#]],
    );
}

#[test]
fn hover_keyword_as_primitive() {
    check(
        r#"
//- /main.rs crate:main deps:std
type F = f$0n(i32) -> i32;
//- /libstd.rs crate:std
/// Docs for prim_fn
mod prim_fn {}
"#,
        expect![[r#"
                *fn*

                ```rust
                fn
                ```

                ---

                Docs for prim_fn
            "#]],
    );
}

#[test]
fn hover_builtin() {
    check(
        r#"
//- /main.rs crate:main deps:std
const _: &str$0 = ""; }

//- /libstd.rs crate:std
/// Docs for prim_str
/// [`foo`](../std/keyword.foo.html)
mod prim_str {}
"#,
        expect![[r#"
            *str*

            ```rust
            str
            ```

            ---

            Docs for prim_str
            [`foo`](https://doc.rust-lang.org/nightly/std/keyword.foo.html)
        "#]],
    );
}

#[test]
fn hover_macro_expanded_function() {
    check(
        r#"
struct S<'a, T>(&'a T);
trait Clone {}
macro_rules! foo {
    () => {
        fn bar<'t, T: Clone + 't>(s: &mut S<'t, T>, t: u32) -> *mut u32 where
            't: 't + 't,
            for<'a> T: Clone + 'a
        { 0 as _ }
    };
}

foo!();

fn main() {
    bar$0;
}
"#,
        expect![[r#"
            *bar*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn bar<'t, T>(s: &mut S<'t, T>, t: u32) -> *mut u32
            where
                T: Clone + 't,
                't: 't + 't,
                for<'a> T: Clone + 'a,
            ```
        "#]],
    )
}

#[test]
fn hover_intra_doc_links() {
    check(
        r#"
pub mod theitem {
    /// This is the item. Cool!
    pub struct TheItem;
}

/// Gives you a [`TheItem$0`].
///
/// [`TheItem`]: theitem::TheItem
pub fn gimme() -> theitem::TheItem {
    theitem::TheItem
}
"#,
        expect![[r#"
            *[`TheItem`]*

            ```rust
            ra_test_fixture::theitem
            ```

            ```rust
            pub struct TheItem
            ```

            ---

            This is the item. Cool!
        "#]],
    );
}

#[test]
fn test_hover_trait_assoc_typealias() {
    check(
        r#"
        fn main() {}

trait T1 {
    type Bar;
    type Baz;
}

struct Foo;

mod t2 {
    pub trait T2 {
        type Bar;
    }
}

use t2::T2;

impl T2 for Foo {
    type Bar = String;
}

impl T1 for Foo {
    type Bar = <Foo as t2::T2>::Ba$0r;
    //                          ^^^ unresolvedReference
}
        "#,
        expect![[r#"
            *Bar*

            ```rust
            ra_test_fixture::t2::T2
            ```

            ```rust
            pub type Bar
            ```
        "#]],
    );
}
#[test]
fn hover_generic_assoc() {
    check(
        r#"
fn foo<T: A>() where T::Assoc$0: {}

trait A {
    type Assoc;
}"#,
        expect![[r#"
            *Assoc*

            ```rust
            ra_test_fixture::A
            ```

            ```rust
            type Assoc
            ```
        "#]],
    );
    check(
        r#"
fn foo<T: A>() {
    let _: <T>::Assoc$0;
}

trait A {
    type Assoc;
}"#,
        expect![[r#"
            *Assoc*

            ```rust
            ra_test_fixture::A
            ```

            ```rust
            type Assoc
            ```
        "#]],
    );
    check(
        r#"
trait A where
    Self::Assoc$0: ,
{
    type Assoc;
}"#,
        expect![[r#"
            *Assoc*

            ```rust
            ra_test_fixture::A
            ```

            ```rust
            type Assoc
            ```
        "#]],
    );
}

#[test]
fn string_shadowed_with_inner_items() {
    check(
        r#"
//- /main.rs crate:main deps:alloc

/// Custom `String` type.
struct String;

fn f() {
    let _: String$0;

    fn inner() {}
}

//- /alloc.rs crate:alloc
#[prelude_import]
pub use string::*;

mod string {
    /// This is `alloc::String`.
    pub struct String;
}
"#,
        expect![[r#"
            *String*

            ```rust
            main
            ```

            ```rust
            struct String
            ```

            ---

            Custom `String` type.
        "#]],
    )
}

#[test]
fn function_doesnt_shadow_crate_in_use_tree() {
    check(
        r#"
//- /main.rs crate:main deps:foo
use foo$0::{foo};

//- /foo.rs crate:foo
pub fn foo() {}
"#,
        expect![[r#"
                *foo*

                ```rust
                extern crate foo
                ```
            "#]],
    )
}

#[test]
fn hover_feature() {
    let (analysis, position) = fixture::position(r#"#![feature(intrinsics$0)]"#);
    analysis
        .hover(
            &HoverConfig { links_in_hover: true, ..HOVER_BASE_CONFIG },
            FileRange { file_id: position.file_id, range: TextRange::empty(position.offset) },
        )
        .unwrap()
        .unwrap();
}

#[test]
fn hover_lint() {
    check(
        r#"#![allow(arithmetic_overflow$0)]"#,
        expect![[r#"
                *arithmetic_overflow*
                ```
                arithmetic_overflow
                ```
                ___

                arithmetic operation overflows
            "#]],
    );
    check(
        r#"#![expect(arithmetic_overflow$0)]"#,
        expect![[r#"
                *arithmetic_overflow*
                ```
                arithmetic_overflow
                ```
                ___

                arithmetic operation overflows
            "#]],
    );
}

#[test]
fn hover_clippy_lint() {
    check(
        r#"#![allow(clippy::almost_swapped$0)]"#,
        expect![[r#"
                *almost_swapped*
                ```
                clippy::almost_swapped
                ```
                ___

                Checks for `foo = bar; bar = foo` sequences.
            "#]],
    );
    check(
        r#"#![expect(clippy::almost_swapped$0)]"#,
        expect![[r#"
                *almost_swapped*
                ```
                clippy::almost_swapped
                ```
                ___

                Checks for `foo = bar; bar = foo` sequences.
            "#]],
    );
}

#[test]
fn hover_attr_path_qualifier() {
    check(
        r#"
//- /foo.rs crate:foo

//- /lib.rs crate:main.rs deps:foo
#[fo$0o::bar()]
struct Foo;
"#,
        expect![[r#"
                *foo*

                ```rust
                extern crate foo
                ```
            "#]],
    )
}

#[test]
fn hover_rename() {
    check(
        r#"
use self as foo$0;
"#,
        expect![[r#"
            *foo*

            ```rust
            extern crate ra_test_fixture
            ```
        "#]],
    );
    check(
        r#"
mod bar {}
use bar::{self as foo$0};
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            mod bar
            ```
        "#]],
    );
    check(
        r#"
mod bar {
    use super as foo$0;
}
"#,
        expect![[r#"
            *foo*

            ```rust
            extern crate ra_test_fixture
            ```
        "#]],
    );
    check(
        r#"
use crate as foo$0;
"#,
        expect![[r#"
            *foo*

            ```rust
            extern crate ra_test_fixture
            ```
        "#]],
    );
}

#[test]
fn hover_attribute_in_macro() {
    check(
        r#"
//- minicore:derive
macro_rules! identity {
    ($struct:item) => {
        $struct
    };
}
#[rustc_builtin_macro]
pub macro Copy {}
identity!{
    #[derive(Copy$0)]
    struct Foo;
}
"#,
        expect![[r#"
            *Copy*

            ```rust
            ra_test_fixture
            ```

            ```rust
            macro Copy
            ```
        "#]],
    );
}

#[test]
fn hover_derive_input() {
    check(
        r#"
//- minicore:derive
#[rustc_builtin_macro]
pub macro Copy {}
#[derive(Copy$0)]
struct Foo;
"#,
        expect![[r#"
            *Copy*

            ```rust
            ra_test_fixture
            ```

            ```rust
            macro Copy
            ```
        "#]],
    );
    check(
        r#"
//- minicore:derive
mod foo {
    #[rustc_builtin_macro]
    pub macro Copy {}
}
#[derive(foo::Copy$0)]
struct Foo;
"#,
        expect![[r#"
            *Copy*

            ```rust
            ra_test_fixture::foo
            ```

            ```rust
            macro Copy
            ```
        "#]],
    );
}

#[test]
fn hover_range_math() {
    check_hover_range(
        r#"
fn f() { let expr = $01 + 2 * 3$0 }
"#,
        expect![[r#"
            ```rust
            i32
            ```"#]],
    );

    check_hover_range(
        r#"
fn f() { let expr = 1 $0+ 2 * $03 }
"#,
        expect![[r#"
            ```rust
            i32
            ```"#]],
    );

    check_hover_range(
        r#"
fn f() { let expr = 1 + $02 * 3$0 }
"#,
        expect![[r#"
            ```rust
            i32
            ```"#]],
    );
}

#[test]
fn hover_range_arrays() {
    check_hover_range(
        r#"
fn f() { let expr = $0[1, 2, 3, 4]$0 }
"#,
        expect![[r#"
            ```rust
            [i32; 4]
            ```"#]],
    );

    check_hover_range(
        r#"
fn f() { let expr = [1, 2, $03, 4]$0 }
"#,
        expect![[r#"
            ```rust
            [i32; 4]
            ```"#]],
    );

    check_hover_range(
        r#"
fn f() { let expr = [1, 2, $03$0, 4] }
"#,
        expect![[r#"
            ```rust
            i32
            ```"#]],
    );
}

#[test]
fn hover_range_functions() {
    check_hover_range(
        r#"
fn f<T>(a: &[T]) { }
fn b() { $0f$0(&[1, 2, 3, 4, 5]); }
"#,
        expect![[r#"
            ```rust
            fn f<i32>(&[i32])
            ```"#]],
    );

    check_hover_range(
        r#"
fn f<T>(a: &[T]) { }
fn b() { f($0&[1, 2, 3, 4, 5]$0); }
"#,
        expect![[r#"
            ```rust
            &[i32; 5]
            ```"#]],
    );
}

#[test]
fn hover_range_shows_nothing_when_invalid() {
    check_hover_range_no_results(
        r#"
fn f<T>(a: &[T]) { }
fn b()$0 { f(&[1, 2, 3, 4, 5]); }$0
"#,
    );

    check_hover_range_no_results(
        r#"
fn f<T>$0(a: &[T]) { }
fn b() { f(&[1, 2, 3,$0 4, 5]); }
"#,
    );

    check_hover_range_no_results(
        r#"
fn $0f() { let expr = [1, 2, 3, 4]$0 }
"#,
    );
}

#[test]
fn hover_range_shows_unit_for_statements() {
    check_hover_range(
        r#"
fn f<T>(a: &[T]) { }
fn b() { $0f(&[1, 2, 3, 4, 5]); }$0
"#,
        expect![[r#"
            ```rust
            ()
            ```"#]],
    );

    check_hover_range(
        r#"
fn f() { let expr$0 = $0[1, 2, 3, 4] }
"#,
        expect![[r#"
            ```rust
            ()
            ```"#]],
    );
}

#[test]
fn hover_range_for_pat() {
    check_hover_range(
        r#"
fn foo() {
    let $0x$0 = 0;
}
"#,
        expect![[r#"
                ```rust
                i32
                ```"#]],
    );

    check_hover_range(
        r#"
fn foo() {
    let $0x$0 = "";
}
"#,
        expect![[r#"
            ```rust
            &'static str
            ```"#]],
    );
}

#[test]
fn hover_range_shows_coercions_if_applicable_expr() {
    check_hover_range(
        r#"
fn foo() {
    let x: &u32 = $0&&&&&0$0;
}
"#,
        expect![[r#"
                ```text
                Type:       &&&&&u32
                Coerced to:     &u32
                ```
            "#]],
    );
    check_hover_range(
        r#"
fn foo() {
    let x: *const u32 = $0&0$0;
}
"#,
        expect![[r#"
                ```text
                Type:             &u32
                Coerced to: *const u32
                ```
            "#]],
    );
}

#[test]
fn hover_range_shows_type_actions() {
    check_actions(
        r#"
struct Foo;
fn foo() {
    let x: &Foo = $0&&&&&Foo$0;
}
"#,
        expect![[r#"
            [
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Foo",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..11,
                                focus_range: 7..10,
                                name: "Foo",
                                kind: Struct,
                                description: "struct Foo",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn hover_try_expr_res() {
    check_hover_range(
        r#"
//- minicore: try, from, result
struct FooError;

fn foo() -> Result<(), FooError> {
    Ok($0Result::<(), FooError>::Ok(())?$0)
}
"#,
        expect![[r#"
                ```rust
                ()
                ```"#]],
    );
    check_hover_range(
        r#"
//- minicore: try, from, result
struct FooError;
struct BarError;

fn foo() -> Result<(), FooError> {
    Ok($0Result::<(), BarError>::Ok(())?$0)
}
"#,
        expect![[r#"
                ```text
                Try Error Type: BarError
                Propagated as:  FooError
                ```
            "#]],
    );
}

#[test]
fn hover_try_expr() {
    check_hover_range(
        r#"
//- minicore: try
struct NotResult<T, U>(T, U);
struct Short;
struct Looooong;

fn foo() -> NotResult<(), Looooong> {
    $0NotResult((), Short)?$0;
}
"#,
        expect![[r#"
                ```text
                Try Target Type:    NotResult<(), Short>
                Propagated as:   NotResult<(), Looooong>
                ```
            "#]],
    );
    check_hover_range(
        r#"
//- minicore: try
struct NotResult<T, U>(T, U);
struct Short;
struct Looooong;

fn foo() -> NotResult<(), Short> {
    $0NotResult((), Looooong)?$0;
}
"#,
        expect![[r#"
                ```text
                Try Target Type: NotResult<(), Looooong>
                Propagated as:      NotResult<(), Short>
                ```
            "#]],
    );
}

#[test]
fn hover_try_expr_option() {
    cov_mark::check!(hover_try_expr_opt_opt);
    check_hover_range(
        r#"
//- minicore: option, try

fn foo() -> Option<()> {
    $0Some(0)?$0;
    None
}
"#,
        expect![[r#"
                ```rust
                i32
                ```"#]],
    );
}

#[test]
fn hover_deref_expr() {
    check_hover_range(
        r#"
//- minicore: deref
use core::ops::Deref;

struct DerefExample<T> {
    value: T
}

impl<T> Deref for DerefExample<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

fn foo() {
    let x = DerefExample { value: 0 };
    let y: i32 = $0*x$0;
}
"#,
        expect![[r#"
                ```text
                Dereferenced from: DerefExample<i32>
                To type:                         i32
                ```
            "#]],
    );
}

#[test]
fn hover_deref_expr_with_coercion() {
    check_hover_range(
        r#"
//- minicore: deref
use core::ops::Deref;

struct DerefExample<T> {
    value: T
}

impl<T> Deref for DerefExample<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

fn foo() {
    let x = DerefExample { value: &&&&&0 };
    let y: &i32 = $0*x$0;
}
"#,
        expect![[r#"
                ```text
                Dereferenced from: DerefExample<&&&&&i32>
                To type:                         &&&&&i32
                Coerced to:                          &i32
                ```
            "#]],
    );
}

#[test]
fn hover_intra_in_macro() {
    check(
        r#"
macro_rules! foo_macro {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        pub struct $name;
    }
}

foo_macro!(
    /// Doc comment for [`Foo$0`]
    Foo
);
"#,
        expect![[r#"
            *[`Foo`]*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub struct Foo
            ```

            ---

            Doc comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/struct.Foo.html)
        "#]],
    );
}

#[test]
fn hover_intra_in_attr() {
    check(
        r#"
#[doc = "Doc comment for [`Foo$0`]"]
pub struct Foo(i32);
"#,
        expect![[r#"
            *[`Foo`]*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub struct Foo(i32)
            ```

            ---

            Doc comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/struct.Foo.html)
        "#]],
    );
}

#[test]
fn hover_intra_inner_attr() {
    check(
        r#"
/// outer comment for [`Foo`]
#[doc = "Doc outer comment for [`Foo`]"]
pub fn Foo {
    //! inner comment for [`Foo$0`]
    #![doc = "Doc inner comment for [`Foo`]"]
}
"#,
        expect![[r#"
            *[`Foo`]*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub fn Foo()
            ```

            ---

            outer comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/fn.Foo.html)
            Doc outer comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/fn.Foo.html)
            inner comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/fn.Foo.html)
            Doc inner comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/fn.Foo.html)
        "#]],
    );

    check(
        r#"
/// outer comment for [`Foo`]
#[doc = "Doc outer comment for [`Foo`]"]
pub mod Foo {
    //! inner comment for [`super::Foo$0`]
    #![doc = "Doc inner comment for [`super::Foo`]"]
}
"#,
        expect![[r#"
            *[`super::Foo`]*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub mod Foo
            ```

            ---

            outer comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/Foo/index.html)
            Doc outer comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/Foo/index.html)
            inner comment for [`super::Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/Foo/index.html)
            Doc inner comment for [`super::Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/Foo/index.html)
        "#]],
    );
}

#[test]
fn hover_intra_outer_attr() {
    check(
        r#"
/// outer comment for [`Foo$0`]
#[doc = "Doc outer comment for [`Foo`]"]
pub fn Foo() {
    //! inner comment for [`Foo`]
    #![doc = "Doc inner comment for [`Foo`]"]
}
"#,
        expect![[r#"
            *[`Foo`]*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub fn Foo()
            ```

            ---

            outer comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/fn.Foo.html)
            Doc outer comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/fn.Foo.html)
            inner comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/fn.Foo.html)
            Doc inner comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/fn.Foo.html)
        "#]],
    );

    check(
        r#"
/// outer comment for [`Foo$0`]
#[doc = "Doc outer comment for [`Foo`]"]
pub mod Foo {
    //! inner comment for [`super::Foo`]
    #![doc = "Doc inner comment for [`super::Foo`]"]
}
"#,
        expect![[r#"
            *[`Foo`]*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub mod Foo
            ```

            ---

            outer comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/Foo/index.html)
            Doc outer comment for [`Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/Foo/index.html)
            inner comment for [`super::Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/Foo/index.html)
            Doc inner comment for [`super::Foo`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/Foo/index.html)
        "#]],
    );
}

#[test]
fn hover_intra_generics() {
    check(
        r#"
/// Doc comment for [`Foo$0<T>`]
pub struct Foo<T>(T);
"#,
        expect![[r#"
            *[`Foo<T>`]*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub struct Foo<T>(T)
            ```

            ---

            Doc comment for [`Foo<T>`](https://docs.rs/ra_test_fixture/*/ra_test_fixture/struct.Foo.html)
        "#]],
    );
}

#[test]
fn hover_inert_attr() {
    check(
        r#"
#[doc$0 = ""]
pub struct Foo;
"#,
        expect![[r##"
            *doc*

            ```rust
            #[doc]
            ```

            ---

            Valid forms are:

            * \#\[doc(hidden|inline|...)\]
            * \#\[doc = string\]
        "##]],
    );
    check(
        r#"
#[allow$0()]
pub struct Foo;
"#,
        expect![[r##"
            *allow*

            ```rust
            #[allow]
            ```

            ---

            Valid forms are:

            * \#\[allow(lint1, lint2, ..., /\*opt\*/ reason = "...")\]
        "##]],
    );
}

#[test]
fn hover_dollar_crate() {
    // $crate should be resolved to the right crate name.

    check(
        r#"
//- /main.rs crate:main deps:dep
dep::m!(KONST$0);
//- /dep.rs crate:dep
#[macro_export]
macro_rules! m {
    ( $name:ident ) => { const $name: $crate::Type = $crate::Type; };
}

pub struct Type;
"#,
        expect![[r#"
            *KONST*

            ```rust
            main
            ```

            ```rust
            const KONST: dep::Type = Type
            ```
        "#]],
    );
}

#[test]
fn hover_record_variant() {
    check(
        r#"
enum Enum {
    RecordV$0 { field: u32 }
}
"#,
        expect![[r#"
            *RecordV*

            ```rust
            ra_test_fixture::Enum
            ```

            ```rust
            RecordV { field: u32, }
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
}

#[test]
fn hover_record_variant_field() {
    check(
        r#"
enum Enum {
    RecordV { field$0: u32 }
}
"#,
        expect![[r#"
            *field*

            ```rust
            ra_test_fixture::Enum::RecordV
            ```

            ```rust
            field: u32
            ```

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
}

#[test]
fn hover_trait_impl_assoc_item_def_doc_forwarding() {
    check(
        r#"
trait T {
    /// Trait docs
    fn func() {}
}
impl T for () {
    fn func$0() {}
}
"#,
        expect![[r#"
            *func*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn func()
            ```

            ---

            Trait docs
        "#]],
    );
}

#[test]
fn hover_trait_show_assoc_items() {
    check_assoc_count(
        0,
        r#"
trait T {}
impl T$0 for () {}
"#,
        expect![[r#"
            *T*

            ```rust
            ra_test_fixture
            ```

            ```rust
            trait T {}
            ```
        "#]],
    );

    check_assoc_count(
        1,
        r#"
trait T {}
impl T$0 for () {}
"#,
        expect![[r#"
            *T*

            ```rust
            ra_test_fixture
            ```

            ```rust
            trait T {}
            ```
        "#]],
    );

    check_assoc_count(
        0,
        r#"
trait T {
    fn func() {}
    const FLAG: i32 = 34;
    type Bar;
}
impl T$0 for () {}
"#,
        expect![[r#"
            *T*

            ```rust
            ra_test_fixture
            ```

            ```rust
            trait T { /* … */ }
            ```
        "#]],
    );

    check_assoc_count(
        2,
        r#"
trait T {
    fn func() {}
    const FLAG: i32 = 34;
    type Bar;
}
impl T$0 for () {}
"#,
        expect![[r#"
            *T*

            ```rust
            ra_test_fixture
            ```

            ```rust
            trait T {
                fn func();
                const FLAG: i32;
                /* … */
            }
            ```
        "#]],
    );

    check_assoc_count(
        3,
        r#"
trait T {
    fn func() {}
    const FLAG: i32 = 34;
    type Bar;
}
impl T$0 for () {}
"#,
        expect![[r#"
            *T*

            ```rust
            ra_test_fixture
            ```

            ```rust
            trait T {
                fn func();
                const FLAG: i32;
                type Bar;
            }
            ```
        "#]],
    );

    check_assoc_count(
        4,
        r#"
trait T {
    fn func() {}
    const FLAG: i32 = 34;
    type Bar;
}
impl T$0 for () {}
"#,
        expect![[r#"
            *T*

            ```rust
            ra_test_fixture
            ```

            ```rust
            trait T {
                fn func();
                const FLAG: i32;
                type Bar;
            }
            ```
        "#]],
    );
}

#[test]
fn hover_ranged_macro_call() {
    check_hover_range(
        r#"
macro_rules! __rust_force_expr {
    ($e:expr) => {
        $e
    };
}
macro_rules! vec {
    ($elem:expr) => {
        __rust_force_expr!($elem)
    };
}

struct Struct;
impl Struct {
    fn foo(self) {}
}

fn f() {
    $0vec![Struct]$0;
}
"#,
        expect![[r#"
            ```rust
            Struct
            ```"#]],
    );
}

#[test]
fn hover_deref() {
    check(
        r#"
//- minicore: deref

struct Struct(usize);

impl core::ops::Deref for Struct {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn f() {
    $0*Struct(0);
}
"#,
        expect![[r#"
            ***

            ```rust
            ra_test_fixture::Struct
            ```

            ```rust
            fn deref(&self) -> &Self::Target
            ```
        "#]],
    );
}

#[test]
fn static_const_macro_expanded_body() {
    check(
        r#"
macro_rules! m {
    () => {
        pub const V: i8 = {
            let e = 123;
            f(e) // Prevent const eval from evaluating this constant, we want to print the body's code.
        };
    };
}
m!();
fn main() { $0V; }
"#,
        expect![[r#"
            *V*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub const V: i8 = {
                let e = 123;
                f(e)
            }
            ```
        "#]],
    );
    check(
        r#"
macro_rules! m {
    () => {
        pub static V: i8 = {
            let e = 123;
        };
    };
}
m!();
fn main() { $0V; }
"#,
        expect![[r#"
            *V*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub static V: i8 = {
                let e = 123;
            }
            ```
        "#]],
    );
}

#[test]
fn hover_rest_pat() {
    check(
        r#"
struct Struct {a: u32, b: u32, c: u8, d: u16};

fn main() {
    let Struct {a, c, .$0.} = Struct {a: 1, b: 2, c: 3, d: 4};
}
"#,
        expect![[r#"
            *..*
            ```rust
            .., b: u32, d: u16
            ```
        "#]],
    );

    check(
        r#"
struct Struct {a: u32, b: u32, c: u8, d: u16};

fn main() {
    let Struct {a, b, c, d, .$0.} = Struct {a: 1, b: 2, c: 3, d: 4};
}
"#,
        expect![[r#"
            *..*
            ```rust
            ..
            ```
        "#]],
    );
}

#[test]
fn hover_underscore_pat() {
    check(
        r#"
fn main() {
    let _$0 = 0;
}
"#,
        expect![[r#"
            *_*
            ```rust
            i32
            ```
        "#]],
    );
    check(
        r#"
fn main() {
    let (_$0,) = (0,);
}
"#,
        expect![[r#"
            *_*
            ```rust
            i32
            ```
        "#]],
    );
}

#[test]
fn hover_underscore_expr() {
    check(
        r#"
fn main() {
    _$0 = 0;
}
"#,
        expect![[r#"
            *_*
            ```rust
            i32
            ```
        "#]],
    );
    check(
        r#"
fn main() {
    (_$0,) = (0,);
}
"#,
        expect![[r#"
            *_*
            ```rust
            i32
            ```
        "#]],
    );
}

#[test]
fn hover_underscore_type() {
    check_hover_no_result(
        r#"
fn main() {
    let x: _$0 = 0;
}
"#,
    );
    check_hover_no_result(
        r#"
fn main() {
    let x: (_$0,) = (0,);
}
"#,
    );
}

#[test]
fn hover_call_parens() {
    check(
        r#"
fn foo() -> i32 {}
fn main() {
    foo($0);
}
"#,
        expect![[r#"
            *)*
            ```rust
            i32
            ```
        "#]],
    );
    check(
        r#"
struct S;
impl S {
    fn foo(self) -> i32 {}
}
fn main() {
    S.foo($0);
}
"#,
        expect![[r#"
            *)*
            ```rust
            i32
            ```
        "#]],
    );
}

#[test]
fn assoc_fn_in_block_local_impl() {
    check(
        r#"
struct S;
mod m {
    const _: () = {
        impl crate::S {
            pub(crate) fn foo() {}
        }
    };
}
fn test() {
    S::foo$0();
}
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::m::S
            ```

            ```rust
            pub(crate) fn foo()
            ```
        "#]],
    );

    check(
        r#"
struct S;
mod m {
    const _: () = {
        const _: () = {
            impl crate::S {
                pub(crate) fn foo() {}
            }
        };
    };
}
fn test() {
    S::foo$0();
}
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::m::S
            ```

            ```rust
            pub(crate) fn foo()
            ```
        "#]],
    );

    check(
        r#"
struct S;
mod m {
    mod inner {
        const _: () = {
            impl crate::S {
                pub(super) fn foo() {}
            }
        };
    }

    fn test() {
        crate::S::foo$0();
    }
}
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::m::inner::S
            ```

            ```rust
            pub(super) fn foo()
            ```
        "#]],
    );
}

#[test]
fn assoc_const_in_block_local_impl() {
    check(
        r#"
struct S;
mod m {
    const _: () = {
        impl crate::S {
            pub(crate) const A: () = ();
        }
    };
}
fn test() {
    S::A$0;
}
"#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture::m::S
            ```

            ```rust
            pub(crate) const A: () = ()
            ```
        "#]],
    );

    check(
        r#"
struct S;
mod m {
    const _: () = {
        const _: () = {
            impl crate::S {
                pub(crate) const A: () = ();
            }
        };
    };
}
fn test() {
    S::A$0;
}
"#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture::m::S
            ```

            ```rust
            pub(crate) const A: () = ()
            ```
        "#]],
    );

    check(
        r#"
struct S;
mod m {
    mod inner {
        const _: () = {
            impl crate::S {
                pub(super) const A: () = ();
            }
        };
    }

    fn test() {
        crate::S::A$0;
    }
}
"#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture::m::inner::S
            ```

            ```rust
            pub(super) const A: () = ()
            ```
        "#]],
    );
}

#[test]
fn field_as_method_call_fallback() {
    check(
        r#"
struct S { f: u32 }
fn test() {
    S { f: 0 }.f$0();
}
"#,
        expect![[r#"
            *f*

            ```rust
            ra_test_fixture::S
            ```

            ```rust
            f: u32
            ```
        "#]],
    );
}

#[test]
fn generic_params_disabled_by_cfg() {
    check(
        r#"
struct S<#[cfg(never)] T>;
fn test() {
    let s$0: S = S;
}
"#,
        expect![[r#"
            *s*

            ```rust
            let s: S
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn format_args_arg() {
    check(
        r#"
//- minicore: fmt
fn test() {
    let foo = 0;
    format_args!("{}", foo$0);
}
"#,
        expect![[r#"
            *foo*

            ```rust
            let foo: i32
            ```
        "#]],
    );
}

#[test]
fn format_args_implicit() {
    check(
        r#"
//- minicore: fmt
fn test() {
let aaaaa = "foo";
format_args!("{aaaaa$0}");
}
"#,
        expect![[r#"
            *aaaaa*

            ```rust
            let aaaaa: &'static str
            ```
        "#]],
    );
}

#[test]
fn format_args_implicit2() {
    check(
        r#"
//- minicore: fmt
fn test() {
let aaaaa = "foo";
format_args!("{$0aaaaa}");
}
"#,
        expect![[r#"
            *aaaaa*

            ```rust
            let aaaaa: &'static str
            ```
        "#]],
    );
}

#[test]
fn format_args_implicit_raw() {
    check(
        r#"
//- minicore: fmt
fn test() {
let aaaaa = "foo";
format_args!(r"{$0aaaaa}");
}
"#,
        expect![[r#"
            *aaaaa*

            ```rust
            let aaaaa: &'static str
            ```
        "#]],
    );
}

#[test]
fn format_args_implicit_nested() {
    check(
        r#"
//- minicore: fmt
macro_rules! foo {
    ($($tt:tt)*) => {
        format_args!($($tt)*)
    }
}
fn test() {
let aaaaa = "foo";
foo!(r"{$0aaaaa}");
}
"#,
        expect![[r#"
            *aaaaa*

            ```rust
            let aaaaa: &'static str
            ```
        "#]],
    );
}

#[test]
fn method_call_without_parens() {
    check(
        r#"
struct S;
impl S {
    fn foo<T>(&self, t: T) {}
}

fn main() {
    S.foo$0;
}
"#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::S
            ```

            ```rust
            fn foo<T>(&self, t: T)
            ```
        "#]],
    );
}

#[test]
fn string_literal() {
    check(
        r#"
fn main() {
    $0"🦀\u{1f980}\\\x41";
}
"#,
        expect![[r#"
            *"🦀\u{1f980}\\\x41"*
            ```rust
            &'static str
            ```
            ___

            value of literal: ` 🦀🦀\A `
        "#]],
    );
    check(
        r#"
fn main() {
    $0r"🦀\u{1f980}\\\x41";
}
"#,
        expect![[r#"
            *r"🦀\u{1f980}\\\x41"*
            ```rust
            &'static str
            ```
            ___

            value of literal: ` 🦀\u{1f980}\\\x41 `
        "#]],
    );
    check(
        r#"
fn main() {
    $0r"🦀\u{1f980}\\\x41


fsdghs";
}
"#,
        expect![[r#"
            *r"🦀\u{1f980}\\\x41


            fsdghs"*
            ```rust
            &'static str
            ```
            ___

            value of literal (truncated up to newline): ` 🦀\u{1f980}\\\x41 `
        "#]],
    );
}

#[test]
fn cstring_literal() {
    check(
        r#"
fn main() {
    $0c"🦀\u{1f980}\\\x41";
}
"#,
        expect![[r#"
            *c"🦀\u{1f980}\\\x41"*
            ```rust
            &'static {unknown}
            ```
            ___

            value of literal: ` 🦀🦀\A `
        "#]],
    );
}

#[test]
fn rawstring_literal() {
    check(
        r#"
fn main() {
    $0r"`[^`]*`";
}"#,
        expect![[r#"
            *r"`[^`]*`"*
            ```rust
            &'static str
            ```
            ___

            value of literal: ```` `[^`]*` ````
        "#]],
    );
    check(
        r#"
fn main() {
    $0r"`";
}"#,
        expect![[r#"
            *r"`"*
            ```rust
            &'static str
            ```
            ___

            value of literal: `` ` ``
        "#]],
    );
    check(
        r#"
fn main() {
    $0r"    ";
}"#,
        expect![[r#"
            *r"    "*
            ```rust
            &'static str
            ```
            ___

            value of literal: `    `
        "#]],
    );
    check(
        r#"
fn main() {
    $0r" Hello World ";

}"#,
        expect![[r#"
            *r" Hello World "*
            ```rust
            &'static str
            ```
            ___

            value of literal: `  Hello World  `
        "#]],
    )
}

#[test]
fn byte_string_literal() {
    check(
        r#"
fn main() {
    $0b"\xF0\x9F\xA6\x80\\";
}
"#,
        expect![[r#"
            *b"\xF0\x9F\xA6\x80\\"*
            ```rust
            &'static [u8; 5]
            ```
            ___

            value of literal: ` [240, 159, 166, 128, 92] `
        "#]],
    );
    check(
        r#"
fn main() {
    $0br"\xF0\x9F\xA6\x80\\";
}
"#,
        expect![[r#"
            *br"\xF0\x9F\xA6\x80\\"*
            ```rust
            &'static [u8; 18]
            ```
            ___

            value of literal: ` [92, 120, 70, 48, 92, 120, 57, 70, 92, 120, 65, 54, 92, 120, 56, 48, 92, 92] `
        "#]],
    );
}

#[test]
fn byte_literal() {
    check(
        r#"
fn main() {
    $0b'\xF0';
}
"#,
        expect![[r#"
            *b'\xF0'*
            ```rust
            u8
            ```
            ___

            value of literal: ` 0xF0 `
        "#]],
    );
    check(
        r#"
fn main() {
    $0b'\\';
}
"#,
        expect![[r#"
            *b'\\'*
            ```rust
            u8
            ```
            ___

            value of literal: ` 0x5C `
        "#]],
    );
}

#[test]
fn char_literal() {
    check(
        r#"
fn main() {
    $0'\x41';
}
"#,
        expect![[r#"
            *'\x41'*
            ```rust
            char
            ```
            ___

            value of literal: ` A `
        "#]],
    );
    check(
        r#"
fn main() {
    $0'\\';
}
"#,
        expect![[r#"
            *'\\'*
            ```rust
            char
            ```
            ___

            value of literal: ` \ `
        "#]],
    );
    check(
        r#"
fn main() {
    $0'\u{1f980}';
}
"#,
        expect![[r#"
            *'\u{1f980}'*
            ```rust
            char
            ```
            ___

            value of literal: ` 🦀 `
        "#]],
    );
}

#[test]
fn float_literal() {
    check(
        r#"
fn main() {
    $01.0;
}
"#,
        expect![[r#"
            *1.0*
            ```rust
            f64
            ```
            ___

            value of literal: ` 1 (bits: 0x3FF0000000000000) `
        "#]],
    );
    check(
        r#"
fn main() {
    $01.0f16;
}
"#,
        expect![[r#"
            *1.0f16*
            ```rust
            f16
            ```
            ___

            value of literal: ` 1 (bits: 0x3C00) `
        "#]],
    );
    check(
        r#"
fn main() {
    $01.0f32;
}
"#,
        expect![[r#"
            *1.0f32*
            ```rust
            f32
            ```
            ___

            value of literal: ` 1 (bits: 0x3F800000) `
        "#]],
    );
    check(
        r#"
fn main() {
    $01.0f128;
}
"#,
        expect![[r#"
            *1.0f128*
            ```rust
            f128
            ```
            ___

            value of literal: ` 1 (bits: 0x3FFF0000000000000000000000000000) `
        "#]],
    );
    check(
        r#"
fn main() {
    $0134e12;
}
"#,
        expect![[r#"
            *134e12*
            ```rust
            f64
            ```
            ___

            value of literal: ` 134000000000000 (bits: 0x42DE77D399980000) `
        "#]],
    );
    check(
        r#"
fn main() {
    $01523527134274733643531312.0;
}
"#,
        expect![[r#"
            *1523527134274733643531312.0*
            ```rust
            f64
            ```
            ___

            value of literal: ` 1523527134274733600000000 (bits: 0x44F429E9249F629B) `
        "#]],
    );
    check(
        r#"
fn main() {
    $00.1ea123;
}
"#,
        expect![[r#"
            *0.1ea123*
            ```rust
            f64
            ```
            ___

            invalid literal: invalid float literal
        "#]],
    );
}

#[test]
fn int_literal() {
    check(
        r#"
fn main() {
    $034325236457856836345234;
}
"#,
        expect![[r#"
            *34325236457856836345234*
            ```rust
            i32
            ```
            ___

            value of literal: ` 34325236457856836345234 (0x744C659178614489D92|0b111010001001100011001011001000101111000011000010100010010001001110110010010) `
        "#]],
    );
    check(
        r#"
fn main() {
    $0134_123424_21;
}
"#,
        expect![[r#"
            *134_123424_21*
            ```rust
            i32
            ```
            ___

            value of literal: ` 13412342421 (0x31F701A95|0b1100011111011100000001101010010101) `
        "#]],
    );
    check(
        r#"
fn main() {
    $00x12423423;
}
"#,
        expect![[r#"
            *0x12423423*
            ```rust
            i32
            ```
            ___

            value of literal: ` 306328611 (0x12423423|0b10010010000100011010000100011) `
        "#]],
    );
    check(
        r#"
fn main() {
    $00b1111_1111;
}
"#,
        expect![[r#"
            *0b1111_1111*
            ```rust
            i32
            ```
            ___

            value of literal: ` 255 (0xFF|0b11111111) `
        "#]],
    );
    check(
        r#"
fn main() {
    $00o12345;
}
"#,
        expect![[r#"
            *0o12345*
            ```rust
            i32
            ```
            ___

            value of literal: ` 5349 (0x14E5|0b1010011100101) `
        "#]],
    );
    check(
        r#"
fn main() {
    $00xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_F;
}
"#,
        expect![[r#"
            *0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_F*
            ```rust
            i32
            ```
            ___

            invalid literal: number too large to fit in target type
        "#]],
    );
}

#[test]
fn notable_local() {
    check(
        r#"
#[doc(notable_trait)]
trait Notable {
    type Assoc;
    type Assoc2;
}

impl Notable for u32 {
    type Assoc = &str;
    type Assoc2 = char;
}
fn main(notable$0: u32) {}
"#,
        expect![[r#"
            *notable*

            ```rust
            notable: u32
            ```

            ---

            Implements notable traits: `Notable<Assoc = &str, Assoc2 = char>`

            ---

            size = 4, align = 4, no Drop
        "#]],
    );
}

#[test]
fn notable_foreign() {
    check(
        r#"
//- minicore: future, iterator
struct S;
#[doc(notable_trait)]
trait Notable {}
impl Notable for S$0 {}
impl core::future::Future for S {
    type Output = u32;
}
impl Iterator for S {
    type Item = S;
}
"#,
        expect![[r#"
            *S*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct S
            ```
        "#]],
    );
}

#[test]
fn extern_items() {
    check(
        r#"
extern "C" {
    static STATIC$0: ();
}
"#,
        expect![[r#"
            *STATIC*

            ```rust
            ra_test_fixture::<extern>
            ```

            ```rust
            static STATIC: ()
            ```
        "#]],
    );
    check(
        r#"
extern "C" {
    fn fun$0();
}
"#,
        expect![[r#"
            *fun*

            ```rust
            ra_test_fixture::<extern>
            ```

            ```rust
            unsafe fn fun()
            ```
        "#]],
    );
    check(
        r#"
extern "C" {
    type Ty$0;
}
"#,
        expect![[r#"
            *Ty*

            ```rust
            ra_test_fixture::<extern>
            ```

            ```rust
            type Ty
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn notable_ranged() {
    check_hover_range(
        r#"
//- minicore: future, iterator
struct S;
#[doc(notable_trait)]
trait Notable {}
impl Notable for S {}
impl core::future::Future for S {
    type Output = u32;
}
impl Iterator for S {
    type Item = S;
}
fn main() {
    $0S$0;
}
"#,
        expect![[r#"
            ```rust
            S
            ```
            ___
            Implements notable traits: `Future<Output = u32>`, `Iterator<Item = S>`, `Notable`"#]],
    );
}

#[test]
fn notable_actions() {
    check_actions(
        r#"
//- minicore: future, iterator
struct S;
struct S2;
#[doc(notable_trait)]
trait Notable {}
impl Notable for S$0 {}
impl core::future::Future for S {
    type Output = u32;
}
impl Iterator for S {
    type Item = S2;
}
"#,
        expect![[r#"
            [
                Implementation(
                    FilePositionWrapper {
                        file_id: FileId(
                            0,
                        ),
                        offset: 7,
                    },
                ),
                GoToType(
                    [
                        HoverGotoTypeData {
                            mod_path: "core::future::Future",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 4294967295..4294967295,
                                focus_range: 4294967295..4294967295,
                                name: "Future",
                                kind: Trait,
                                container_name: "future",
                                description: "pub trait Future",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "core::iter::traits::iterator::Iterator",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    1,
                                ),
                                full_range: 4294967295..4294967295,
                                focus_range: 4294967295..4294967295,
                                name: "Iterator",
                                kind: Trait,
                                container_name: "iterator",
                                description: "pub trait Iterator",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::Notable",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 21..59,
                                focus_range: 49..56,
                                name: "Notable",
                                kind: Trait,
                                description: "trait Notable",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "ra_test_fixture::S2",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 10..20,
                                focus_range: 17..19,
                                name: "S2",
                                kind: Struct,
                                description: "struct S2",
                            },
                        },
                    ],
                ),
            ]
        "#]],
    );
}

#[test]
fn hover_lifetime_regression_16963() {
    check(
        r#"
struct Pedro$0<'a> {
    hola: &'a str
}
"#,
        expect![[r#"
            *Pedro*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Pedro<'a> {
                hola: &'a str,
            }
            ```

            ---

            size = 16 (0x10), align = 8, niches = 1, no Drop
        "#]],
    )
}

#[test]
fn hover_impl_trait_arg_self() {
    check(
        r#"
trait T<Rhs = Self> {}
fn main(a$0: impl T) {}
"#,
        expect![[r#"
            *a*

            ```rust
            a: impl T + ?Sized
            ```

            ---

            type param may need Drop
        "#]],
    );
}

#[test]
fn hover_struct_default_arg_self() {
    check(
        r#"
struct T<Rhs = Self> {}
fn main(a$0: T) {}
"#,
        expect![[r#"
            *a*

            ```rust
            a: T
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn hover_fn_with_impl_trait_arg() {
    check(
        r#"
trait Foo {}
impl Foo for bool {}
fn bar<const WIDTH: u8>(_: impl Foo) {}
fn test() {
    let f = bar::<3>;
    f$0(true);
}
"#,
        expect![[r#"
            *f*

            ```rust
            let f: fn bar<3>(bool)
            ```
        "#]],
    );
}

#[test]
fn issue_17871() {
    check(
        r#"
trait T {
    fn f<A>();
}

struct S {}
impl T for S {
    fn f<A>() {}
}

fn main() {
    let x$0 = S::f::<i32>;
}
"#,
        expect![[r#"
            *x*

            ```rust
            let x: fn f<S, i32>()
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn raw_keyword_different_editions() {
    check(
        r#"
//- /lib1.rs crate:with_edition_2015 edition:2015
pub fn dyn() {}

//- /lib2.rs crate:with_edition_2018 edition:2018 deps:with_edition_2015 new_source_root:local
fn foo() {
    with_edition_2015::r#dyn$0();
}
    "#,
        expect![[r#"
            *r#dyn*

            ```rust
            with_edition_2015
            ```

            ```rust
            pub fn r#dyn()
            ```
        "#]],
    );

    check(
        r#"
//- /lib1.rs crate:with_edition_2018 edition:2018
pub fn r#dyn() {}

//- /lib2.rs crate:with_edition_2015 edition:2015 deps:with_edition_2018 new_source_root:local
fn foo() {
    with_edition_2018::dyn$0();
}
    "#,
        expect![[r#"
            *dyn*

            ```rust
            with_edition_2018
            ```

            ```rust
            pub fn dyn()
            ```
        "#]],
    );

    check(
        r#"
//- /lib1.rs crate:escaping_needlessly edition:2015
pub fn r#dyn() {}

//- /lib2.rs crate:dependent edition:2015 deps:escaping_needlessly new_source_root:local
fn foo() {
    escaping_needlessly::dyn$0();
}
    "#,
        expect![[r#"
            *dyn*

            ```rust
            escaping_needlessly
            ```

            ```rust
            pub fn dyn()
            ```
        "#]],
    );
}

#[test]
fn test_hover_function_with_pat_param() {
    check(
        r#"fn test_1$0((start_range, end_range): (u32, u32), a: i32) {}"#,
        expect![[r#"
            *test_1*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn test_1((start_range, end_range): (u32, u32), a: i32)
            ```
        "#]],
    );

    // Test case with tuple pattern and mutable parameters
    check(
        r#"fn test_2$0((mut x, y): (i32, i32)) {}"#,
        expect![[r#"
            *test_2*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn test_2((mut x, y): (i32, i32))
            ```
        "#]],
    );

    // Test case with a pattern in a reference type
    check(
        r#"fn test_3$0(&(a, b): &(i32, i32)) {}"#,
        expect![[r#"
            *test_3*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn test_3(&(a, b): &(i32, i32))
            ```
        "#]],
    );

    // Test case with complex pattern (struct destructuring)
    check(
        r#"struct Point { x: i32, y: i32 } fn test_4$0(Point { x, y }: Point) {}"#,
        expect![[r#"
            *test_4*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn test_4(Point { x, y }: Point)
            ```
        "#]],
    );

    // Test case with a nested pattern
    check(
        r#"fn test_5$0(((a, b), c): ((i32, i32), i32)) {}"#,
        expect![[r#"
            *test_5*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn test_5(((a, b), c): ((i32, i32), i32))
            ```
        "#]],
    );

    // Test case with an unused variable in the pattern
    check(
        r#"fn test_6$0((_, y): (i32, i64)) {}"#,
        expect![[r#"
            *test_6*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn test_6((_, y): (i32, i64))
            ```
        "#]],
    );

    // Test case with a complex pattern involving both tuple and struct
    check(
        r#"struct Foo { a: i32, b: i32 } fn test_7$0((x, Foo { a, b }): (i32, Foo)) {}"#,
        expect![[r#"
            *test_7*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn test_7((x, Foo { a, b }): (i32, Foo))
            ```
        "#]],
    );

    // Test case with Enum and Or pattern
    check(
        r#"enum MyEnum { A(i32), B(i32) } fn test_8$0((MyEnum::A(x) | MyEnum::B(x)): MyEnum) {}"#,
        expect![[r#"
            *test_8*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn test_8((MyEnum::A(x) | MyEnum::B(x)): MyEnum)
            ```
        "#]],
    );

    // Test case with a pattern as a function parameter
    check(
        r#"struct Foo { a: i32, b: i32 } fn test_9$0(Foo { a, b }: Foo) {}"#,
        expect![[r#"
            *test_9*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn test_9(Foo { a, b }: Foo)
            ```
        "#]],
    );

    // Test case with a pattern as a function parameter with a different name
    check(
        r#"struct Foo { a: i32, b: i32 } fn test_10$0(Foo { a, b: b1 }: Foo) {}"#,
        expect![[r#"
            *test_10*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn test_10(Foo { a, b: b1 }: Foo)
            ```
        "#]],
    );

    // Test case with a pattern as a function parameter with annotations
    check(
        r#"struct Foo { a: i32, b: i32 } fn test_10$0(Foo { a, b: mut b }: Foo) {}"#,
        expect![[r#"
            *test_10*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn test_10(Foo { a, b: mut b }: Foo)
            ```
        "#]],
    );
}

#[test]
fn hover_path_inside_block_scope() {
    check(
        r#"
mod m {
    const _: () = {
        mod m2 {
            const C$0: () = ();
        }
    };
}
"#,
        expect![[r#"
            *C*

            ```rust
            ra_test_fixture::m::m2
            ```

            ```rust
            const C: () = ()
            ```
        "#]],
    );
}

#[test]
fn regression_18238() {
    check(
        r#"
macro_rules! foo {
    ($name:ident) => {
        pub static $name = Foo::new(|| {
            $crate;
        });
    };
}

foo!(BAR_$0);
"#,
        expect![[r#"
            *BAR_*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub static BAR_: {error} = Foo::new(||{
                crate;
            })
            ```
        "#]],
    );
}

#[test]
fn type_alias_without_docs() {
    // Simple.
    check(
        r#"
/// Docs for B
struct B;

type A$0 = B;
"#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture
            ```

            ```rust
            type A = B
            ```

            ---

            size = 0, align = 1, no Drop

            ---

            *This is the documentation for* `struct B`

            Docs for B
        "#]],
    );

    // Nested.
    check(
        r#"
/// Docs for C
struct C;

type B = C;

type A$0 = B;
"#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture
            ```

            ```rust
            type A = B
            ```

            ---

            size = 0, align = 1, no Drop

            ---

            *This is the documentation for* `struct C`

            Docs for C
        "#]],
    );

    // Showing the docs for aliased struct instead of intermediate type.
    check(
        r#"
/// Docs for C
struct C;

/// Docs for B
type B = C;

type A$0 = B;
"#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture
            ```

            ```rust
            type A = B
            ```

            ---

            size = 0, align = 1, no Drop

            ---

            *This is the documentation for* `struct C`

            Docs for C
        "#]],
    );

    // No docs found.
    check(
        r#"
struct C;

type B = C;

type A$0 = B;
"#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture
            ```

            ```rust
            type A = B
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );

    // Multiple nested crate.
    check(
        r#"
//- /lib.rs crate:c
/// Docs for C
pub struct C;

//- /lib.rs crate:b deps:c
pub use c::C;
pub type B = C;

//- /lib.rs crate:a deps:b
pub use b::B;
pub type A = B;

//- /main.rs crate:main deps:a
use a::A$0;
"#,
        expect![[r#"
            *A*

            ```rust
            a
            ```

            ```rust
            pub type A = B
            ```

            ---

            *This is the documentation for* `pub struct C`

            Docs for C
        "#]],
    );
}

#[test]
fn dyn_compat() {
    check(
        r#"
trait Compat$0 {}
"#,
        expect![[r#"
            *Compat*

            ```rust
            ra_test_fixture
            ```

            ```rust
            trait Compat
            ```

            ---

            Is dyn-compatible
        "#]],
    );
    check(
        r#"
trait UnCompat$0 {
    fn f<T>() {}
}
"#,
        expect![[r#"
            *UnCompat*

            ```rust
            ra_test_fixture
            ```

            ```rust
            trait UnCompat
            ```

            ---

            Is not dyn-compatible due to having a method `f` that is not dispatchable due to missing a receiver
        "#]],
    );
    check(
        r#"
trait UnCompat {
    fn f<T>() {}
}
fn f<T: UnCompat$0>
"#,
        expect![[r#"
            *UnCompat*

            ```rust
            ra_test_fixture
            ```

            ```rust
            trait UnCompat
            ```
        "#]],
    );
}

#[test]
fn issue_18613() {
    check(
        r#"
fn main() {
    struct S<T, D = bool>();
    let x$0 = S::<()>;
}"#,
        expect![[r#"
            *x*

            ```rust
            let x: fn S<()>() -> S<()>
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );

    check(
        r#"
pub struct Global;
pub struct Box<T, A = Global>(T, A);

impl<T> Box<T> {
    pub fn new(x: T) -> Self { loop {} }
}

pub struct String;

fn main() {
    let box_value$0 = Box::<String>new();
}
"#,
        expect![[r#"
            *box_value*

            ```rust
            let box_value: fn Box<String>(String, Global) -> Box<String>
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );

    check(
        r#"
//- minicore: eq
pub struct RandomState;
pub struct HashMap<K, V, S = RandomState>(K, V, S);

impl<K, V> HashMap<K, V, RandomState> {
    pub fn new() -> HashMap<K, V, RandomState> {
        loop {}
    }
}

impl<K, V, S> PartialEq for HashMap<K, V, S> {
    fn eq(&self, other: &HashMap<K, V, S>) -> bool {
        false
    }
}

fn main() {
    let s$0 = HashMap::<_, u64>::ne;
}
"#,
        expect![[r#"
            *s*

            ```rust
            let s: fn ne<HashMap<{unknown}, u64>>(&HashMap<{unknown}, u64>, &HashMap<{unknown}, u64>) -> bool
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
}

#[test]
fn subst_fn() {
    check(
        r#"
struct Foo<T>(T);
impl<T> Foo<T> {
    fn foo<U>(v: T, u: U) {}
}

fn bar() {
    Foo::fo$0o(123, false);
}
        "#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            impl<T> Foo<T>
            fn foo<U>(v: T, u: U)
            ```

            ---

            `T` = `i32`, `U` = `bool`
        "#]],
    );
    check(
        r#"
fn foo<T>(v: T) {}

fn bar() {
    fo$0o(123);
}
        "#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            fn foo<T>(v: T)
            ```

            ---

            `T` = `i32`
        "#]],
    );
}

#[test]
fn subst_record_constructor() {
    check(
        r#"
struct Foo<T> { field: T }

fn bar() {
    let v = $0Foo { field: 123 };
}
        "#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo<T> {
                field: T,
            }
            ```

            ---

            `T` = `i32`
        "#]],
    );
    check(
        r#"
struct Foo<T> { field: T }

fn bar() {
    let v = Foo { field: 123 };
    let $0Foo { field: _ } = v;
}
        "#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo<T> {
                field: T,
            }
            ```

            ---

            `T` = `i32`
        "#]],
    );
}

#[test]
fn subst_method_call() {
    check(
        r#"
struct Foo<T>(T);

impl<U> Foo<U> {
    fn bar<T>(self, v: T) {}
}

fn baz() {
    Foo(123).bar$0("hello");
}
    "#,
        expect![[r#"
            *bar*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            impl<U> Foo<U>
            fn bar<T>(self, v: T)
            ```

            ---

            `U` = `i32`, `T` = `&'static str`
        "#]],
    );
}

#[test]
fn subst_type_alias_do_not_work() {
    // It is very hard to support subst for type aliases properly in all places because they are eagerly evaluated.
    // We can show the user the subst for the underlying type instead but that'll be very confusing.
    check(
        r#"
struct Foo<T, U> { a: T, b: U }
type Alias<T> = Foo<T, i32>;

fn foo() {
    let _ = Alias$0 { a: true, b: 123 };
}
    "#,
        expect![[r#"
            *Alias*

            ```rust
            ra_test_fixture
            ```

            ```rust
            type Alias<T> = Foo<T, i32>
            ```
        "#]],
    );
}

#[test]
fn subst_self() {
    check(
        r#"
trait Trait<T> {
    fn foo<U>(&self, v: U) {}
}
struct Struct<T>(T);
impl<T> Trait<i64> for Struct<T> {}

fn bar() {
    Struct(123).foo$0(true);
}
    "#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::Trait
            ```

            ```rust
            trait Trait<T>
            fn foo<U>(&self, v: U)
            ```

            ---

            `Self` = `Struct<i32>`, `T` = `i64`, `U` = `bool`
        "#]],
    );
}

#[test]
fn subst_with_lifetimes_and_consts() {
    check(
        r#"
struct Foo<'a, const N: usize, T>(&[T; N]);

impl<'a, T, const N: usize> Foo<'a, N, T> {
    fn foo<'b, const Z: u32, U>(&self, v: U) {}
}

fn bar() {
    Foo(&[1i8]).fo$0o::<456, _>("");
}
    "#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            impl<'a, T, const N: usize> Foo<'a, N, T>
            fn foo<'b, const Z: u32, U>(&self, v: U)
            ```

            ---

            `T` = `i8`, `U` = `&'static str`
        "#]],
    );
}

#[test]
fn subst_field() {
    check(
        r#"
struct Foo<T> { field: T }

fn bar() {
    let v = Foo { $0field: 123 };
}
    "#,
        expect![[r#"
            *field*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            field: T
            ```

            ---

            `T` = `i32`
        "#]],
    );
    check(
        r#"
struct Foo<T> { field: T }

fn bar() {
    let field = 123;
    let v = Foo { field$0 };
}
    "#,
        expect![[r#"
            *field*

            ```rust
            let field: i32
            ```

            ---

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            field: T
            ```

            ---

            `T` = `i32`
        "#]],
    );
    check(
        r#"
struct Foo<T> { field: T }

fn bar() {
    let v = Foo { field: 123 };
    let Foo { field$0 } = v;
}
    "#,
        expect![[r#"
            *field*

            ```rust
            let field: i32
            ```

            ---

            size = 4, align = 4, no Drop

            ---

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            field: T
            ```

            ---

            type param may need Drop

            ---

            `T` = `i32`
        "#]],
    );
    check(
        r#"
struct Foo<T> { field: T }

fn bar() {
    let v = Foo { field: 123 };
    let Foo { field$0: _ } = v;
}
    "#,
        expect![[r#"
            *field*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            field: T
            ```

            ---

            `T` = `i32`
        "#]],
    );
    check(
        r#"
struct Foo<T> { field: T }

fn bar() {
    let v = Foo { field: 123 };
    let _ = (&v).$0field;
}
    "#,
        expect![[r#"
            *field*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            field: T
            ```

            ---

            `T` = `i32`
        "#]],
    );
    check(
        r#"
struct Foo<T>(T);

fn bar() {
    let v = Foo(123);
    let _ = v.$00;
}
    "#,
        expect![[r#"
            *0*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            0: T
            ```

            ---

            `T` = `i32`
        "#]],
    );
}

#[test]
fn i128_max() {
    check(
        r#"
//- /core.rs library crate:core
#![rustc_coherence_is_core]
impl u128 {
    pub const MAX: Self = 340_282_366_920_938_463_463_374_607_431_768_211_455u128;
}
impl i128 {
    pub const MAX: Self = (u128::MAX >> 1) as Self;
}

//- /foo.rs crate:foo deps:core
fn foo() {
    let _ = i128::MAX$0;
}
        "#,
        expect![
            r#"
            *MAX*

            ```rust
            core
            ```

            ```rust
            pub const MAX: Self = 170141183460469231731687303715884105727 (0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)
            ```
        "#
        ],
    );
}

#[test]
fn test_runnables_with_snapshot_tests() {
    check_actions(
        r#"
//- /lib.rs crate:foo deps:expect_test,insta,snapbox
use expect_test::expect;
use insta::assert_debug_snapshot;
use snapbox::Assert;

#[test]
fn test$0() {
    let actual = "new25";
    expect!["new25"].assert_eq(&actual);
    Assert::new()
        .action_env("SNAPSHOTS")
        .eq(actual, snapbox::str!["new25"]);
    assert_debug_snapshot!(actual);
}

//- /lib.rs crate:expect_test
struct Expect;

impl Expect {
    fn assert_eq(&self, actual: &str) {}
}

#[macro_export]
macro_rules! expect {
    ($e:expr) => Expect; // dummy
}

//- /lib.rs crate:insta
#[macro_export]
macro_rules! assert_debug_snapshot {
    ($e:expr) => {}; // dummy
}

//- /lib.rs crate:snapbox
pub struct Assert;

impl Assert {
    pub fn new() -> Self { Assert }

    pub fn action_env(&self, env: &str) -> &Self { self }

    pub fn eq(&self, actual: &str, expected: &str) {}
}

#[macro_export]
macro_rules! str {
    ($e:expr) => ""; // dummy
}
        "#,
        expect![[r#"
            [
                Reference(
                    FilePositionWrapper {
                        file_id: FileId(
                            0,
                        ),
                        offset: 92,
                    },
                ),
                Runnable(
                    Runnable {
                        use_name_in_title: false,
                        nav: NavigationTarget {
                            file_id: FileId(
                                0,
                            ),
                            full_range: 81..301,
                            focus_range: 92..96,
                            name: "test",
                            kind: Function,
                        },
                        kind: Test {
                            test_id: Path(
                                "test",
                            ),
                            attr: TestAttr {
                                ignore: false,
                            },
                        },
                        cfg: None,
                        update_test: UpdateTest {
                            expect_test: true,
                            insta: true,
                            snapbox: true,
                        },
                    },
                ),
            ]
        "#]],
    );
}

#[test]
fn drop_glue() {
    check(
        r#"
struct NoDrop$0;
    "#,
        expect![[r#"
            *NoDrop*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct NoDrop
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
    check(
        r#"
//- minicore: drop
struct NeedsDrop$0;
impl Drop for NeedsDrop {
    fn drop(&mut self) {}
}
    "#,
        expect![[r#"
            *NeedsDrop*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct NeedsDrop
            ```

            ---

            size = 0, align = 1, impl Drop
        "#]],
    );
    check(
        r#"
//- minicore: manually_drop, drop
struct NeedsDrop;
impl Drop for NeedsDrop {
    fn drop(&mut self) {}
}
type NoDrop$0 = core::mem::ManuallyDrop<NeedsDrop>;
    "#,
        expect![[r#"
            *NoDrop*

            ```rust
            ra_test_fixture
            ```

            ```rust
            type NoDrop = core::mem::ManuallyDrop<NeedsDrop>
            ```

            ---

            size = 0, align = 1, no Drop
        "#]],
    );
    check(
        r#"
//- minicore: drop
struct NeedsDrop;
impl Drop for NeedsDrop {
    fn drop(&mut self) {}
}
struct DropField$0 {
    _x: i32,
    _y: NeedsDrop,
}
    "#,
        expect![[r#"
            *DropField*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct DropField {
                _x: i32,
                _y: NeedsDrop,
            }
            ```

            ---

            size = 4, align = 4, needs Drop
        "#]],
    );
    check(
        r#"
//- minicore: sized
type Foo$0 = impl Sized;
    "#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            type Foo = impl Sized
            ```

            ---

            needs Drop
        "#]],
    );
    check(
        r#"
//- minicore: drop
struct NeedsDrop;
impl Drop for NeedsDrop {
    fn drop(&mut self) {}
}
enum Enum {
    A$0(&'static str),
    B(NeedsDrop)
}
    "#,
        expect![[r#"
            *A*

            ```rust
            ra_test_fixture::Enum
            ```

            ```rust
            A(&'static str)
            ```

            ---

            size = 16 (0x10), align = 8, niches = 1, no Drop
        "#]],
    );
    check(
        r#"
struct Foo$0<T>(T);
    "#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo<T>(T)
            ```

            ---

            type param may need Drop
        "#]],
    );
    check(
        r#"
//- minicore: copy
struct Foo$0<T: Copy>(T);
    "#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo<T>(T)
            where
                T: Copy,
            ```

            ---

            no Drop
        "#]],
    );
    check(
        r#"
//- minicore: copy
trait Trait {
    type Assoc: Copy;
}
struct Foo$0<T: Trait>(T::Assoc);
    "#,
        expect![[r#"
            *Foo*

            ```rust
            ra_test_fixture
            ```

            ```rust
            struct Foo<T>(<T as Trait>::Assoc)
            where
                T: Trait,
            ```

            ---

            no Drop
        "#]],
    );
    check(
        r#"
#[rustc_coherence_is_core]

#[lang = "manually_drop"]
#[repr(transparent)]
pub struct ManuallyDrop$0<T: ?Sized> {
    value: T,
}
    "#,
        expect![[r#"
            *ManuallyDrop*

            ```rust
            ra_test_fixture
            ```

            ```rust
            pub struct ManuallyDrop<T>
            where
                T: ?Sized,
            {
                value: T,
            }
            ```

            ---

            no Drop
        "#]],
    );
}

#[test]
fn projection_const() {
    // This uses two crates, which have *no* relation between them, to test another thing:
    // `render_const_scalar()` used to just use the last crate for the trait env, which will
    // fail in this scenario.
    check(
        r#"
//- /foo.rs crate:foo
pub trait PublicFlags {
    type Internal;
}

pub struct NoteDialects(<NoteDialects as PublicFlags>::Internal);

impl NoteDialects {
    pub const CLAP$0: Self = Self(InternalBitFlags);
}

pub struct InternalBitFlags;

impl PublicFlags for NoteDialects {
    type Internal = InternalBitFlags;
}
//- /bar.rs crate:bar
    "#,
        expect![[r#"
            *CLAP*

            ```rust
            foo::NoteDialects
            ```

            ```rust
            pub const CLAP: Self = NoteDialects(InternalBitFlags)
            ```
        "#]],
    );
}

#[test]
fn bounds_from_container_do_not_panic() {
    check(
        r#"
//- minicore: copy
struct Foo<T>(T);

impl<T: Copy> Foo<T> {
    fn foo<U: Copy>(&self, _u: U) {}
}

fn bar(v: &Foo<i32>) {
    v.$0foo(1u32);
}
    "#,
        expect![[r#"
            *foo*

            ```rust
            ra_test_fixture::Foo
            ```

            ```rust
            impl<T> Foo<T>
            fn foo<U>(&self, _u: U)
            where
                U: Copy,
                // Bounds from impl:
                T: Copy,
            ```

            ---

            `T` = `i32`, `U` = `u32`
        "#]],
    );
}

#[test]
fn extra_lifetime_param_on_trait_method_subst() {
    check(
        r#"
struct AudioFormat;

trait ValueEnum {
    fn to_possible_value(&self);
}

impl ValueEnum for AudioFormat {
    fn to_possible_value<'a>(&'a self) {}
}

fn main() {
    ValueEnum::to_possible_value$0(&AudioFormat);
}
    "#,
        expect![[r#"
            *to_possible_value*

            ```rust
            ra_test_fixture::AudioFormat
            ```

            ```rust
            fn to_possible_value<'a>(&'a self)
            ```
        "#]],
    );
}
