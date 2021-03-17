use crate::RootDatabase;
use base_db::{fixture::ChangeFixture, FilePosition};
use expect_test::{expect, Expect};
use test_utils::RangeOrOffset;

/// Creates analysis from a multi-file fixture, returns positions marked with $0.
pub(crate) fn position(ra_fixture: &str) -> (RootDatabase, FilePosition) {
    let change_fixture = ChangeFixture::parse(ra_fixture);
    let mut database = RootDatabase::default();
    database.apply_change(change_fixture.change);
    let (file_id, range_or_offset) = change_fixture.file_position.expect("expected a marker ($0)");
    let offset = match range_or_offset {
        RangeOrOffset::Range(_) => panic!(),
        RangeOrOffset::Offset(it) => it,
    };
    (database, FilePosition { file_id, offset })
}

fn check(ra_fixture: &str, expect: Expect) {
    let (db, position) = position(ra_fixture);
    let call_info = crate::call_info::call_info(&db, position);
    let actual = match call_info {
        Some(call_info) => {
            let docs = match &call_info.doc {
                None => "".to_string(),
                Some(docs) => format!("{}\n------\n", docs.as_str()),
            };
            let params = call_info
                .parameter_labels()
                .enumerate()
                .map(|(i, param)| {
                    if Some(i) == call_info.active_parameter {
                        format!("<{}>", param)
                    } else {
                        param.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}{}\n({})\n", docs, call_info.signature, params)
        }
        None => String::new(),
    };
    expect.assert_eq(&actual);
}

#[test]
fn test_fn_signature_two_args() {
    check(
        r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo($03, ); }
"#,
        expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                (<x: u32>, y: u32)
            "#]],
    );
    check(
        r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3$0, ); }
"#,
        expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                (<x: u32>, y: u32)
            "#]],
    );
    check(
        r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3,$0 ); }
"#,
        expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                (x: u32, <y: u32>)
            "#]],
    );
    check(
        r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3, $0); }
"#,
        expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                (x: u32, <y: u32>)
            "#]],
    );
}

#[test]
fn test_fn_signature_two_args_empty() {
    check(
        r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo($0); }
"#,
        expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                (<x: u32>, y: u32)
            "#]],
    );
}

#[test]
fn test_fn_signature_two_args_first_generics() {
    check(
        r#"
fn foo<T, U: Copy + Display>(x: T, y: U) -> u32
    where T: Copy + Display, U: Debug
{ x + y }

fn bar() { foo($03, ); }
"#,
        expect![[r#"
                fn foo(x: i32, y: {unknown}) -> u32
                (<x: i32>, y: {unknown})
            "#]],
    );
}

#[test]
fn test_fn_signature_no_params() {
    check(
        r#"
fn foo<T>() -> T where T: Copy + Display {}
fn bar() { foo($0); }
"#,
        expect![[r#"
                fn foo() -> {unknown}
                ()
            "#]],
    );
}

#[test]
fn test_fn_signature_for_impl() {
    check(
        r#"
struct F;
impl F { pub fn new() { } }
fn bar() {
    let _ : F = F::new($0);
}
"#,
        expect![[r#"
                fn new()
                ()
            "#]],
    );
}

#[test]
fn test_fn_signature_for_method_self() {
    check(
        r#"
struct S;
impl S { pub fn do_it(&self) {} }

fn bar() {
    let s: S = S;
    s.do_it($0);
}
"#,
        expect![[r#"
                fn do_it(&self)
                ()
            "#]],
    );
}

#[test]
fn test_fn_signature_for_method_with_arg() {
    check(
        r#"
struct S;
impl S {
    fn foo(&self, x: i32) {}
}

fn main() { S.foo($0); }
"#,
        expect![[r#"
                fn foo(&self, x: i32)
                (<x: i32>)
            "#]],
    );
}

#[test]
fn test_fn_signature_for_method_with_arg_as_assoc_fn() {
    check(
        r#"
struct S;
impl S {
    fn foo(&self, x: i32) {}
}

fn main() { S::foo($0); }
"#,
        expect![[r#"
                fn foo(self: &S, x: i32)
                (<self: &S>, x: i32)
            "#]],
    );
}

#[test]
fn test_fn_signature_with_docs_simple() {
    check(
        r#"
/// test
// non-doc-comment
fn foo(j: u32) -> u32 {
    j
}

fn bar() {
    let _ = foo($0);
}
"#,
        expect![[r#"
            test
            ------
            fn foo(j: u32) -> u32
            (<j: u32>)
        "#]],
    );
}

#[test]
fn test_fn_signature_with_docs() {
    check(
        r#"
/// Adds one to the number given.
///
/// # Examples
///
/// ```
/// let five = 5;
///
/// assert_eq!(6, my_crate::add_one(5));
/// ```
pub fn add_one(x: i32) -> i32 {
    x + 1
}

pub fn do() {
    add_one($0
}"#,
        expect![[r##"
            Adds one to the number given.

            # Examples

            ```
            let five = 5;

            assert_eq!(6, my_crate::add_one(5));
            ```
            ------
            fn add_one(x: i32) -> i32
            (<x: i32>)
        "##]],
    );
}

#[test]
fn test_fn_signature_with_docs_impl() {
    check(
        r#"
struct addr;
impl addr {
    /// Adds one to the number given.
    ///
    /// # Examples
    ///
    /// ```
    /// let five = 5;
    ///
    /// assert_eq!(6, my_crate::add_one(5));
    /// ```
    pub fn add_one(x: i32) -> i32 {
        x + 1
    }
}

pub fn do_it() {
    addr {};
    addr::add_one($0);
}
"#,
        expect![[r##"
            Adds one to the number given.

            # Examples

            ```
            let five = 5;

            assert_eq!(6, my_crate::add_one(5));
            ```
            ------
            fn add_one(x: i32) -> i32
            (<x: i32>)
        "##]],
    );
}

#[test]
fn test_fn_signature_with_docs_from_actix() {
    check(
        r#"
struct WriteHandler<E>;

impl<E> WriteHandler<E> {
    /// Method is called when writer emits error.
    ///
    /// If this method returns `ErrorAction::Continue` writer processing
    /// continues otherwise stream processing stops.
    fn error(&mut self, err: E, ctx: &mut Self::Context) -> Running {
        Running::Stop
    }

    /// Method is called when writer finishes.
    ///
    /// By default this method stops actor's `Context`.
    fn finished(&mut self, ctx: &mut Self::Context) {
        ctx.stop()
    }
}

pub fn foo(mut r: WriteHandler<()>) {
    r.finished($0);
}
"#,
        expect![[r#"
            Method is called when writer finishes.

            By default this method stops actor's `Context`.
            ------
            fn finished(&mut self, ctx: &mut {unknown})
            (<ctx: &mut {unknown}>)
        "#]],
    );
}

#[test]
fn call_info_bad_offset() {
    cov_mark::check!(call_info_bad_offset);
    check(
        r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo $0 (3, ); }
"#,
        expect![[""]],
    );
}

#[test]
fn test_nested_method_in_lambda() {
    check(
        r#"
struct Foo;
impl Foo { fn bar(&self, _: u32) { } }

fn bar(_: u32) { }

fn main() {
    let foo = Foo;
    std::thread::spawn(move || foo.bar($0));
}
"#,
        expect![[r#"
                fn bar(&self, _: u32)
                (<_: u32>)
            "#]],
    );
}

#[test]
fn works_for_tuple_structs() {
    check(
        r#"
/// A cool tuple struct
struct S(u32, i32);
fn main() {
    let s = S(0, $0);
}
"#,
        expect![[r#"
            A cool tuple struct
            ------
            struct S(u32, i32)
            (u32, <i32>)
        "#]],
    );
}

#[test]
fn generic_struct() {
    check(
        r#"
struct S<T>(T);
fn main() {
    let s = S($0);
}
"#,
        expect![[r#"
                struct S({unknown})
                (<{unknown}>)
            "#]],
    );
}

#[test]
fn works_for_enum_variants() {
    check(
        r#"
enum E {
    /// A Variant
    A(i32),
    /// Another
    B,
    /// And C
    C { a: i32, b: i32 }
}

fn main() {
    let a = E::A($0);
}
"#,
        expect![[r#"
            A Variant
            ------
            enum E::A(i32)
            (<i32>)
        "#]],
    );
}

#[test]
fn cant_call_struct_record() {
    check(
        r#"
struct S { x: u32, y: i32 }
fn main() {
    let s = S($0);
}
"#,
        expect![[""]],
    );
}

#[test]
fn cant_call_enum_record() {
    check(
        r#"
enum E {
    /// A Variant
    A(i32),
    /// Another
    B,
    /// And C
    C { a: i32, b: i32 }
}

fn main() {
    let a = E::C($0);
}
"#,
        expect![[""]],
    );
}

#[test]
fn fn_signature_for_call_in_macro() {
    check(
        r#"
macro_rules! id { ($($tt:tt)*) => { $($tt)* } }
fn foo() { }
id! {
    fn bar() { foo($0); }
}
"#,
        expect![[r#"
                fn foo()
                ()
            "#]],
    );
}

#[test]
fn call_info_for_lambdas() {
    check(
        r#"
struct S;
fn foo(s: S) -> i32 { 92 }
fn main() {
    (|s| foo(s))($0)
}
        "#,
        expect![[r#"
                (S) -> i32
                (<S>)
            "#]],
    )
}

#[test]
fn call_info_for_fn_ptr() {
    check(
        r#"
fn main(f: fn(i32, f64) -> char) {
    f(0, $0)
}
        "#,
        expect![[r#"
                (i32, f64) -> char
                (i32, <f64>)
            "#]],
    )
}
