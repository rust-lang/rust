use expect_test::{expect, Expect};

use crate::tests::{check_edit, completion_list};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual);
}

#[test]
fn with_default_impl() {
    check(
        r#"
//- minicore: default
struct Struct { foo: u32, bar: usize }

impl Default for Struct {
    fn default() -> Self {
        Struct {
            foo: 0,
            bar: 0,
        }
    }
}

fn foo() {
    let other = Struct {
        foo: 5,
        $0
    };
}
"#,
        expect![[r#"
            fd ..Default::default()
            fd bar                  usize
        "#]],
    );
}

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
        "#]],
    );
}

#[test]
fn functional_update() {
    // FIXME: This should filter out all completions that do not have the type `Foo`
    check(
        r#"
struct Foo { foo1: u32, foo2: u32 }

fn main() {
    let thing = 1;
    let foo = Foo { foo1: 0, foo2: 0 };
    let foo2 = Foo { thing, ..$0 }
}
"#,
        expect![[r#"
            kw unsafe
            kw match
            kw while
            kw while let
            kw loop
            kw if
            kw if let
            kw for
            kw true
            kw false
            kw return
            kw self
            kw super
            kw crate
            lc foo       Foo
            lc thing     i32
            st Foo
            fn main()    fn()
            bt u32
        "#]],
    );
}

#[test]
fn default_completion_edit() {
    check_edit(
        "..Default::default()",
        r#"
//- minicore: default
struct Struct { foo: u32, bar: usize }

impl Default for Struct {
    fn default() -> Self {}
}

fn foo() {
    let other = Struct {
        foo: 5,
        .$0
    };
}
"#,
        r#"
struct Struct { foo: u32, bar: usize }

impl Default for Struct {
    fn default() -> Self {}
}

fn foo() {
    let other = Struct {
        foo: 5,
        ..Default::default()
    };
}
"#,
    );
    check_edit(
        "..Default::default()",
        r#"
//- minicore: default
struct Struct { foo: u32, bar: usize }

impl Default for Struct {
    fn default() -> Self {}
}

fn foo() {
    let other = Struct {
        foo: 5,
        $0
    };
}
"#,
        r#"
struct Struct { foo: u32, bar: usize }

impl Default for Struct {
    fn default() -> Self {}
}

fn foo() {
    let other = Struct {
        foo: 5,
        ..Default::default()
    };
}
"#,
    );
}
