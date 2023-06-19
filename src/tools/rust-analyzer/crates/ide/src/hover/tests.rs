use expect_test::{expect, Expect};
use ide_db::base_db::{FileLoader, FileRange};
use syntax::TextRange;

use crate::{
    fixture, HoverConfig, HoverDocFormat, MemoryLayoutHoverConfig, MemoryLayoutHoverRenderKind,
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
};

fn check_hover_no_result(ra_fixture: &str) {
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
fn check(ra_fixture: &str, expect: Expect) {
    let (analysis, position) = fixture::position(ra_fixture);
    let hover = analysis
        .hover(
            &HoverConfig { links_in_hover: true, ..HOVER_BASE_CONFIG },
            FileRange { file_id: position.file_id, range: TextRange::empty(position.offset) },
        )
        .unwrap()
        .unwrap();

    let content = analysis.db.file_text(position.file_id);
    let hovered_element = &content[hover.range];

    let actual = format!("*{hovered_element}*\n{}\n", hover.info.markup);
    expect.assert_eq(&actual)
}

fn check_hover_no_links(ra_fixture: &str, expect: Expect) {
    let (analysis, position) = fixture::position(ra_fixture);
    let hover = analysis
        .hover(
            &HOVER_BASE_CONFIG,
            FileRange { file_id: position.file_id, range: TextRange::empty(position.offset) },
        )
        .unwrap()
        .unwrap();

    let content = analysis.db.file_text(position.file_id);
    let hovered_element = &content[hover.range];

    let actual = format!("*{hovered_element}*\n{}\n", hover.info.markup);
    expect.assert_eq(&actual)
}

fn check_hover_no_memory_layout(ra_fixture: &str, expect: Expect) {
    let (analysis, position) = fixture::position(ra_fixture);
    let hover = analysis
        .hover(
            &HoverConfig { memory_layout: None, ..HOVER_BASE_CONFIG },
            FileRange { file_id: position.file_id, range: TextRange::empty(position.offset) },
        )
        .unwrap()
        .unwrap();

    let content = analysis.db.file_text(position.file_id);
    let hovered_element = &content[hover.range];

    let actual = format!("*{hovered_element}*\n{}\n", hover.info.markup);
    expect.assert_eq(&actual)
}

fn check_hover_no_markdown(ra_fixture: &str, expect: Expect) {
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

    let content = analysis.db.file_text(position.file_id);
    let hovered_element = &content[hover.range];

    let actual = format!("*{hovered_element}*\n{}\n", hover.info.markup);
    expect.assert_eq(&actual)
}

fn check_actions(ra_fixture: &str, expect: Expect) {
    let (analysis, file_id, position) = fixture::range_or_position(ra_fixture);
    let hover = analysis
        .hover(
            &HoverConfig { links_in_hover: true, ..HOVER_BASE_CONFIG },
            FileRange { file_id, range: position.range_or_empty() },
        )
        .unwrap()
        .unwrap();
    expect.assert_debug_eq(&hover.info.actions)
}

fn check_hover_range(ra_fixture: &str, expect: Expect) {
    let (analysis, range) = fixture::range(ra_fixture);
    let hover = analysis.hover(&HOVER_BASE_CONFIG, range).unwrap().unwrap();
    expect.assert_eq(hover.info.markup.as_str())
}

fn check_hover_range_actions(ra_fixture: &str, expect: Expect) {
    let (analysis, range) = fixture::range(ra_fixture);
    let hover = analysis
        .hover(&HoverConfig { links_in_hover: true, ..HOVER_BASE_CONFIG }, range)
        .unwrap()
        .unwrap();
    expect.assert_debug_eq(&hover.info.actions);
}

fn check_hover_range_no_results(ra_fixture: &str) {
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
            let local: i32 // size = 4, align = 4
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
            test::module
            ```

            ```rust
            fn abc()
            ```

            ---

            Inner
            ---

            ```rust
            test
            ```

            ```rust
            fn abc()
            ```

            ---

            Outer
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
            test

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
            {closure#0} // size = 8, align = 8, niches = 1
            impl Fn(i32) -> i32
            ```

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
            {closure#0} // size = 0, align = 1
            impl Fn(i32) -> i32
            ```

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
            {closure#0} // size = 16 (0x10), align = 8, niches = 1
            impl FnOnce()
            ```

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
            {closure#0} // size = 8, align = 8, niches = 1
            impl FnMut()
            ```

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
            {closure#0} // size = 8, align = 8, niches = 1
            impl FnOnce() -> S2
            ```
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
                            mod_path: "test::S2",
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
                            mod_path: "test::S",
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
            let mut iter: Iter<Scan<OtherStruct<OtherStruct<i32>>, impl Fn(&mut u32, &u32, &mut u32) -> Option<u32>, u32>> // size = 8, align = 4
            ```
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
                test
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
            test
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
                test::m
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
            test
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
            test
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
                test
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
                test
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
        expect![[r##"
                *foo*

                ```rust
                test
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
            "##]],
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
        expect![[r##"
                *foo*

                ```rust
                test
                ```

                ```rust
                pub fn foo(_: &Path)
                ```

                ---

                Raw string doc attr
            "##]],
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
            test::Foo
            ```

            ```rust
            field_a: u8 // size = 1, align = 1, offset = 4
            ```
        "#]],
    );
}

#[test]
fn hover_shows_struct_field_info() {
    // Hovering over the field when instantiating
    check(
        r#"
struct Foo { field_a: u32 }

fn main() {
    let foo = Foo { field_a$0: 0, };
}
"#,
        expect![[r#"
            *field_a*

            ```rust
            test::Foo
            ```

            ```rust
            field_a: u32 // size = 4, align = 4, offset = 0
            ```
        "#]],
    );

    // Hovering over the field in the definition
    check(
        r#"
struct Foo { field_a$0: u32 }

fn main() {
    let foo = Foo { field_a: 0 };
}
"#,
        expect![[r#"
            *field_a*

            ```rust
            test::Foo
            ```

            ```rust
            field_a: u32 // size = 4, align = 4, offset = 0
            ```
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
            test
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
            test
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
            test
            ```

            ```rust
            static foo: u32 = 456
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
            test
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
            let zz: Test<i32> // size = 8, align = 4
            ```
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
                test::Option
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
            let bar: Option<i32> // size = 4, align = 4
            ```
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
                test::Option
                ```

                ```rust
                None
                ```

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
                test::Option
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
            foo: i32 // size = 4, align = 4
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
            foo: i32 // size = 4, align = 4
            ```
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
            foo: i32 // size = 4, align = 4
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
            foo: i32 // size = 4, align = 4
            ```
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
            let foo_test: Thing // size = 4, align = 4
            ```
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
                test::wrapper::Thing
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
            test
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
                test
                ```

                ```rust
                struct Thing
                ```
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
                test
                ```

                ```rust
                struct Thing
                ```
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
                test
                ```

                ```rust
                enum Thing
                ```
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
                test
                ```

                ```rust
                enum Thing
                ```
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
            let x: i32 // size = 4, align = 4
            ```
        "#]],
    )
}

#[test]
fn test_hover_macro_invocation() {
    check(
        r#"
macro_rules! foo { () => {} }

fn f() { fo$0o!(); }
"#,
        expect![[r#"
                *foo*

                ```rust
                test
                ```

                ```rust
                macro_rules! foo
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
                test
                ```

                ```rust
                macro foo
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
                test
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
                test
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
            bar: u32 // size = 4, align = 4
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
            bar: u32 // size = 4, align = 4
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
            test
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
                test
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
                            mod_path: "test::Bar",
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
fn test_hover_through_literal_string_in_builtin_macro() {
    check_hover_no_result(
        r#"
            #[rustc_builtin_macro]
            macro_rules! format {}

            fn foo() {
                format!("hel$0lo {}", 0);
            }
"#,
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
                test
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
                test
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
                test
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
                test::m
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
                test
                ```

                ```rust
                fn foo(a: i32, b: i32) -> i32
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
            test
            ```

            ```rust
            type foo = fn(a: i32, b: i32) -> i32 // size = 8, align = 8, niches = 1
            ```
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
            test
            ```

            ```rust
            type foo = fn(i32, i32) -> i32 // size = 8, align = 8, niches = 1
            ```
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
                        FilePosition {
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
                extern crate std
                ```

                ---

                Standard library for this test

                Printed?
                abc123
            "#]],
    );
    check(
        r#"
//- /main.rs crate:main deps:std
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
                extern crate std
                ```

                ---

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
                test
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
            test
            ```

            ```rust
            struct Bar // size = 0, align = 1
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
            test
            ```

            ```rust
            struct Bar // size = 0, align = 1
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
            test
            ```

            ```rust
            struct Bar // size = 0, align = 1
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
            test
            ```

            ```rust
            pub struct Bar // size = 0, align = 1
            ```

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
            test
            ```

            ```rust
            pub struct Bar // size = 0, align = 1
            ```

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
                test
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
            test::Foo
            ```

            ```rust
            Variant1(u8, u16) // size = 4, align = 2
            ```
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
            test
            ```

            ```rust
            enum Foo // size = 16 (0x10), align = 8, niches = 254
            ```
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
            test::Foo
            ```

            ```rust
            field_a: u8
            ```
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
            {closure#0}
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
                test::Bar
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
                test::Bar
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
            test
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
                        FilePosition {
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
                        FilePosition {
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
                        FilePosition {
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
                        FilePosition {
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
                        FilePosition {
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
                    FilePosition {
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
                                mod_path: "test::S",
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
                                mod_path: "test::S",
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
                            HoverGotoTypeData {
                                mod_path: "test::Arg",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..16,
                                    focus_range: 7..10,
                                    name: "Arg",
                                    kind: Struct,
                                    description: "struct Arg",
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
                                mod_path: "test::S",
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
                            HoverGotoTypeData {
                                mod_path: "test::Arg",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..16,
                                    focus_range: 7..10,
                                    name: "Arg",
                                    kind: Struct,
                                    description: "struct Arg",
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
                            mod_path: "test::A",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 0..14,
                                focus_range: 7..8,
                                name: "A",
                                kind: Struct,
                                description: "struct A",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "test::B",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 15..29,
                                focus_range: 22..23,
                                name: "B",
                                kind: Struct,
                                description: "struct B",
                            },
                        },
                        HoverGotoTypeData {
                            mod_path: "test::M::C",
                            nav: NavigationTarget {
                                file_id: FileId(
                                    0,
                                ),
                                full_range: 42..60,
                                focus_range: 53..54,
                                name: "C",
                                kind: Struct,
                                container_name: "M",
                                description: "pub struct C",
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
                                mod_path: "test::Foo",
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
                                mod_path: "test::Foo",
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
                                mod_path: "test::S",
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
                                mod_path: "test::Foo",
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
                                mod_path: "test::Bar",
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
                                mod_path: "test::Foo",
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
                                mod_path: "test::Bar",
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
                                mod_path: "test::S1",
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
                                mod_path: "test::S2",
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
                                mod_path: "test::Foo",
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
                                mod_path: "test::Foo",
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
                                mod_path: "test::Bar",
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
                                mod_path: "test::S",
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
                                full_range: 21..69,
                                focus_range: 60..66,
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
                                mod_path: "test::Foo",
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
                                mod_path: "test::S",
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
trait Foo {}
struct S;
impl Foo for S {}

struct B<T>{}
fn foo() -> B<dyn Foo> {}

fn main() { let s$0t = foo(); }
"#,
        expect![[r#"
                [
                    GoToType(
                        [
                            HoverGotoTypeData {
                                mod_path: "test::B",
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 42..55,
                                    focus_range: 49..50,
                                    name: "B",
                                    kind: Struct,
                                    description: "struct B<T>",
                                },
                            },
                            HoverGotoTypeData {
                                mod_path: "test::Foo",
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
                                mod_path: "test::Foo",
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
                                mod_path: "test::Foo",
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
                                mod_path: "test::S",
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
                                mod_path: "test::ImplTrait",
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
                                mod_path: "test::B",
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
                                mod_path: "test::DynTrait",
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
                                mod_path: "test::S",
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
                                mod_path: "test::Foo",
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
                                mod_path: "test::Bar",
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
                                mod_path: "test::Foo",
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
                                mod_path: "test::Foo",
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
            f: &i32 // size = 8, align = 8, niches = 1
            ```
            ---

            ```rust
            test::S
            ```

            ```rust
            f: i32 // size = 4, align = 4, offset = 0
            ```
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
            test
            ```

            ```rust
            struct ST<const C: usize, T = Foo>
            ```
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
            let value: Const<1> // size = 0, align = 1
            ```
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
            let value: Const<0> // size = 0, align = 1
            ```
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
            let value: Const<-1> // size = 0, align = 1
            ```
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
            let value: Const<true> // size = 0, align = 1
            ```
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
            let value: Const<'🦀'> // size = 0, align = 1
            ```
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
            self: &Foo // size = 8, align = 8, niches = 1
            ```
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
            self: Arc<Foo> // size = 0, align = 1
            ```
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
                test
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
                test
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
                test
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
            'lifetime
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
                T
                ```
                "#]],
    );
    // lifetimes bounds arent being tracked yet
    check(
        r#"
//- minicore: sized
struct Foo<T>(T);
impl<T: 'static> Foo<T$0> {}
"#,
        expect![[r#"
                *T*

                ```rust
                T
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
impl<T: Trait> Foo<T$0> {}
"#,
        expect![[r#"
                *T*

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
impl<T: Trait + ?Sized> Foo<T$0> {}
"#,
        expect![[r#"
                *T*

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
                    T
                    ```
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
                    T
                    ```
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
                    T: ?Sized
                    ```
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
                    T: Trait
                    ```
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
                    T: Trait
                    ```
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
                    T: Trait + ?Sized
                    ```
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
                    T
                    ```
                "#]],
        );
        check(
            r#"
//- minicore: sized
trait Trait {}
fn foo<T$0: Sized + ?Sized + Sized + Trait>() {}
"#,
            expect![[r#"
                    *T*

                    ```rust
                    T: Trait
                    ```
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
            test
            ```

            ```rust
            type Foo2 = Foo<2> // size = 0, align = 1
            ```
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
            test::E
            ```

            ```rust
            A = 8 // size = 1, align = 1
            ```

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
            test::E
            ```

            ```rust
            A = 12 (0xC) // size = 1, align = 1
            ```

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
            test::E
            ```

            ```rust
            B = 2 // size = 1, align = 1
            ```

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
            test::E
            ```

            ```rust
            B = 5 // size = 1, align = 1
            ```

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
        test
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
        test
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
            test
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
            test
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
            test
            ```

            ```rust
            const B: bool = false
            ```

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
            test
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
            test
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
            test
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
            test
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
            test
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
            test
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
            test
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
            test
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
            test
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
            test
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
            test
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
            test
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
            test
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
            test
            ```

            ```rust
            const FOO: f64 = 3.3201169227365472
            ```
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
            test
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
            test
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
            test
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
//- minicore: fmt, coerce_unsized, builtin_impls
use core::fmt::Debug;

const FOO$0: &dyn Debug = &2i32;
"#,
        expect![[r#"
            *FOO*

            ```rust
            test
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
            test
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
            test
            ```

            ```rust
            const FOO: &[i32; 5] = &[12, 12, 12, 12, 12]
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
            test
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
            test
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
            test
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
            test
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
            test
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
            test
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
            test
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
            test
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
            test::E
            ```

            ```rust
            A = 3 // size = 0, align = 1
            ```

            ---

            This is a doc
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
            let tile4: [u32; 8] // size = 32 (0x20), align = 4
            ```
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
                test
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
                test::foo
                ```

                ```rust
                mod bar
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
                test
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
            test::theitem
            ```

            ```rust
            pub struct TheItem // size = 0, align = 1
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
test::t2
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
                test
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
                test
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
                test
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
            struct String // size = 0, align = 1
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
    check(
        r#"#![feature(box_syntax$0)]"#,
        expect![[r##"
                *box_syntax*
                ```
                box_syntax
                ```
                ___

                # `box_syntax`

                The tracking issue for this feature is: [#49733]

                [#49733]: https://github.com/rust-lang/rust/issues/49733

                See also [`box_patterns`](box-patterns.md)

                ------------------------

                Currently the only stable way to create a `Box` is via the `Box::new` method.
                Also it is not possible in stable Rust to destructure a `Box` in a match
                pattern. The unstable `box` keyword can be used to create a `Box`. An example
                usage would be:

                ```rust
                #![feature(box_syntax)]

                fn main() {
                    let b = box 5;
                }
                ```

            "##]],
    )
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
    )
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
    )
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
                extern crate test
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
                test
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
                extern crate test
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
                extern crate test
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
            test
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
            test
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
            test::foo
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
                &str
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
                                mod_path: "test::Foo",
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
            test
            ```

            ```rust
            pub struct Foo // size = 0, align = 1
            ```

            ---

            Doc comment for [`Foo`](https://docs.rs/test/*/test/struct.Foo.html)
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
            test
            ```

            ```rust
            pub struct Foo // size = 4, align = 4
            ```

            ---

            Doc comment for [`Foo`](https://docs.rs/test/*/test/struct.Foo.html)
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
            test::Enum
            ```

            ```rust
            RecordV { field: u32 } // size = 4, align = 4
            ```
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
            test::RecordV
            ```

            ```rust
            field: u32 // size = 4, align = 4
            ```
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
            test
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
            test::Struct
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
            test
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
            test
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
            test::S
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
            test::S
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
            test::S
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
            test
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
            test
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
            test
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
            test::S
            ```

            ```rust
            f: u32 // size = 4, align = 4, offset = 0
            ```
        "#]],
    );
}
