use expect_test::expect;

use crate::tests::check;

use super::check_edit;

#[test]
fn without_default_impl() {
    check(
        r#"
struct Struct { foo: u32, bar: usize }

fn foo() {
    let other = Struct {
        foo: 5,
        $0
    };
}
"#,
        expect![[r#"
            fd bar usize
        "#]],
    );
}

#[test]
fn record_pattern_field() {
    check(
        r#"
struct Struct { foo: u32, bar: u32 }

fn foo(s: Struct) {
    match s {
        Struct { foo, $0: 92 } => (),
    }
}
"#,
        expect![[r#"
            fd bar u32
            kw mut
            kw ref
        "#]],
    );
}

#[test]
fn record_pattern_field_enum() {
    check(
        r#"
//- minicore:result
enum Baz { Foo, Bar }

fn foo(baz: Baz) {
    match baz {
        Baz::Foo => (),
        $0
    }
}
"#,
        expect![[r#"
            en Baz
            en Result
            md core
            ev Err
            ev Ok
            bn Baz::Bar Baz::Bar$0
            bn Baz::Foo Baz::Foo$0
            bn Err(…)    Err($1)$0
            bn Ok(…)      Ok($1)$0
            kw mut
            kw ref
        "#]],
    );

    check(
        r#"
//- minicore:result
enum Baz { Foo, Bar }

fn foo(baz: Baz) {
    use Baz::*;
    match baz {
        Foo => (),
        $0
    }
}
 "#,
        expect![[r#"
            en Baz
            en Result
            md core
            ev Bar
            ev Err
            ev Foo
            ev Ok
            bn Bar        Bar$0
            bn Err(…) Err($1)$0
            bn Foo        Foo$0
            bn Ok(…)   Ok($1)$0
            kw mut
            kw ref
        "#]],
    );
}

#[test]
fn pattern_enum_variant() {
    check(
        r#"
enum Enum { Variant { foo: u32, bar: u32 } }
fn foo(e: Enum) {
    match e {
        Enum::Variant { foo, $0 } => (),
    }
}
"#,
        expect![[r#"
            fd bar u32
            kw mut
            kw ref
        "#]],
    );
}

#[test]
fn record_literal_field_in_macro() {
    check(
        r#"
macro_rules! m { ($e:expr) => { $e } }
struct Struct { field: u32 }
fn foo() {
    m!(Struct { fie$0 })
}
"#,
        expect![[r#"
            fd field u32
        "#]],
    );
}

#[test]
fn record_pattern_field_in_macro() {
    check(
        r"
macro_rules! m { ($e:expr) => { $e } }
struct Struct { field: u32 }

fn foo(f: Struct) {
    m!(match f {
        Struct { f$0: 92 } => (),
    })
}
",
        expect![[r#"
            fd field u32
            kw mut
            kw ref
        "#]],
    );
}

#[test]
fn in_functional_update() {
    cov_mark::check!(functional_update);

    check(
        r#"
//- minicore:default
struct Foo { foo1: u32, foo2: u32 }
impl Default for Foo {
    fn default() -> Self { loop {} }
}

fn main() {
    let thing = 1;
    let foo = Foo { foo1: 0, foo2: 0 };
    let foo2 = Foo { thing, ..$0 }
}
"#,
        expect![[r#"
            fd ..Default::default()
            fn main()                          fn()
            lc foo                              Foo
            lc thing                            i32
            md core
            st Foo                              Foo
            st Foo {…} Foo { foo1: u32, foo2: u32 }
            tt Default
            bt u32                              u32
            kw crate::
            kw self::
            ex Foo::default()
            ex foo
        "#]],
    );
    check(
        r#"
//- minicore:default
struct Foo { foo1: u32, foo2: u32 }
impl Default for Foo {
    fn default() -> Self { loop {} }
}

fn main() {
    let thing = 1;
    let foo = Foo { foo1: 0, foo2: 0 };
    let foo2 = Foo { thing, ..Default::$0 }
}
"#,
        expect![[r#"
            fn default() (as Default) fn() -> Self
        "#]],
    );
}

#[test]
fn functional_update_no_dot() {
    cov_mark::check!(functional_update_field);
    // FIXME: This should filter out all completions that do not have the type `Foo`
    check(
        r#"
//- minicore:default
struct Foo { foo1: u32, foo2: u32 }
impl Default for Foo {
    fn default() -> Self { loop {} }
}

fn main() {
    let thing = 1;
    let foo = Foo { foo1: 0, foo2: 0 };
    let foo2 = Foo { thing, $0 }
}
"#,
        expect![[r#"
            fd ..Default::default()
            fd foo1             u32
            fd foo2             u32
        "#]],
    );
}

#[test]
fn functional_update_one_dot() {
    cov_mark::check!(functional_update_one_dot);
    check(
        r#"
//- minicore:default
struct Foo { foo1: u32, foo2: u32 }
impl Default for Foo {
    fn default() -> Self { loop {} }
}

fn main() {
    let thing = 1;
    let foo = Foo { foo1: 0, foo2: 0 };
    let foo2 = Foo { thing, .$0 }
}
"#,
        expect![[r#"
            fd ..Default::default()
            sn ..
        "#]],
    );
}

#[test]
fn empty_union_literal() {
    check(
        r#"
union Union { foo: u32, bar: f32 }

fn foo() {
    let other = Union {
        $0
    };
}
        "#,
        expect![[r#"
            fd bar f32
            fd foo u32
        "#]],
    )
}

#[test]
fn dont_suggest_additional_union_fields() {
    check(
        r#"
union Union { foo: u32, bar: f32 }

fn foo() {
    let other = Union {
        foo: 1,
        $0
    };
}
        "#,
        expect![[r#""#]],
    )
}

#[test]
fn add_space_after_vis_kw() {
    check_edit(
        "pub(crate)",
        r"
pub(crate) struct S {
    $0
}
",
        r#"
pub(crate) struct S {
    pub(crate) $0
}
"#,
    );

    check_edit(
        "pub",
        r"
pub struct S {
    $0
}
",
        r#"
pub struct S {
    pub $0
}
"#,
    );

    check_edit(
        "pub(super)",
        r"
pub(super) struct S {
    $0
}
",
        r#"
pub(super) struct S {
    pub(super) $0
}
"#,
    );
}
