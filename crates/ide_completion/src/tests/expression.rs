//! Completion tests for expressions.
use expect_test::{expect, Expect};

use crate::tests::{completion_list, BASE_ITEMS_FIXTURE};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{}{}", BASE_ITEMS_FIXTURE, ra_fixture));
    expect.assert_eq(&actual)
}

fn check_empty(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual);
}

#[test]
fn complete_literal_struct_with_a_private_field() {
    // `FooDesc.bar` is private, the completion should not be triggered.
    check(
        r#"
mod _69latrick {
    pub struct FooDesc { pub six: bool, pub neuf: Vec<String>, bar: bool }
    pub fn create_foo(foo_desc: &FooDesc) -> () { () }
}

fn baz() {
    use _69latrick::*;

    let foo = create_foo(&$0);
}
            "#,
        // This should not contain `FooDesc {…}`.
        expect![[r##"
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
            kw mut
            kw return
            kw self
            kw super
            kw crate
            st FooDesc
            fn create_foo(…) fn(&FooDesc)
            bt u32
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            fn baz()         fn()
            st Unit
            md _69latrick
            ma makro!(…)     #[macro_export] macro_rules! makro
            fn function()    fn()
            sc STATIC
            un Union
            ev TupleV(…)     (u32)
            ct CONST
        "##]],
    )
}

#[test]
fn completes_various_bindings() {
    check_empty(
        r#"
fn func(param0 @ (param1, param2): (i32, i32)) {
    let letlocal = 92;
    if let ifletlocal = 100 {
        match 0 {
            matcharm => 1 + $0,
            otherwise => (),
        }
    }
    let letlocal2 = 44;
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
            lc matcharm   i32
            lc ifletlocal i32
            lc letlocal   i32
            lc param0     (i32, i32)
            lc param1     i32
            lc param2     i32
            fn func(…)    fn((i32, i32))
            bt u32
        "#]],
    );
}

#[test]
fn completes_all_the_things_in_fn_body() {
    cov_mark::check!(unqualified_skip_lifetime_completion);
    check(
        r#"
use non_existant::Unresolved;
mod qualified { pub enum Enum { Variant } }

impl Unit {
    fn foo<'lifetime, TypeParam, const CONST_PARAM: usize>(self) {
        fn local_func() {}
        $0
    }
}
"#,
        // `self` is in here twice, once as the module, once as the local
        expect![[r##"
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
            kw match
            kw while
            kw while let
            kw loop
            kw if
            kw if let
            kw for
            kw true
            kw false
            kw let
            kw return
            sn pd
            sn ppd
            kw self
            kw super
            kw crate
            fn local_func() fn()
            bt u32
            lc self         Unit
            tp TypeParam
            cp CONST_PARAM
            sp Self
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            md qualified
            ma makro!(…)    #[macro_export] macro_rules! makro
            ?? Unresolved
            fn function()   fn()
            sc STATIC
            un Union
            ev TupleV(…)    (u32)
            ct CONST
            me self.foo()   fn(self)
        "##]],
    );
    check(
        r#"
use non_existant::Unresolved;
mod qualified { pub enum Enum { Variant } }

impl Unit {
    fn foo<'lifetime, TypeParam, const CONST_PARAM: usize>(self) {
        fn local_func() {}
        self::$0
    }
}
"#,
        expect![[r##"
            tt Trait
            en Enum
            st Record
            st Tuple
            md module
            st Unit
            md qualified
            ma makro!(…)  #[macro_export] macro_rules! makro
            ?? Unresolved
            fn function() fn()
            sc STATIC
            un Union
            ev TupleV(…)  (u32)
            ct CONST
        "##]],
    );
}

#[test]
fn complete_in_block() {
    check_empty(
        r#"
    fn foo() {
        if true {
            $0
        }
    }
"#,
        expect![[r#"
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
            kw match
            kw while
            kw while let
            kw loop
            kw if
            kw if let
            kw for
            kw true
            kw false
            kw let
            kw return
            sn pd
            sn ppd
            kw self
            kw super
            kw crate
            fn foo()     fn()
            bt u32
        "#]],
    )
}

#[test]
fn complete_after_if_expr() {
    check_empty(
        r#"
    fn foo() {
        if true {}
        $0
    }
"#,
        expect![[r#"
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
            kw match
            kw while
            kw while let
            kw loop
            kw if
            kw if let
            kw for
            kw true
            kw false
            kw let
            kw else
            kw else if
            kw return
            sn pd
            sn ppd
            kw self
            kw super
            kw crate
            fn foo()     fn()
            bt u32
        "#]],
    )
}

#[test]
fn complete_in_match_arm() {
    check_empty(
        r#"
    fn foo() {
        match () {
            () => $0
        }
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
            fn foo()     fn()
            bt u32
        "#]],
    )
}

#[test]
fn completes_in_loop_ctx() {
    check_empty(
        r"fn my() { loop { $0 } }",
        expect![[r#"
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
            kw match
            kw while
            kw while let
            kw loop
            kw if
            kw if let
            kw for
            kw true
            kw false
            kw let
            kw continue
            kw break
            kw return
            sn pd
            sn ppd
            kw self
            kw super
            kw crate
            fn my()      fn()
            bt u32
        "#]],
    );
}

#[test]
fn completes_in_let_initializer() {
    check_empty(
        r#"fn main() { let _ = $0 }"#,
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
            fn main()    fn()
            bt u32
        "#]],
    )
}

#[test]
fn struct_initializer_field_expr() {
    check_empty(
        r#"
struct Foo {
    pub f: i32,
}
fn foo() {
    Foo {
        f: $0
    }
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
            st Foo
            fn foo()     fn()
            bt u32
        "#]],
    );
}

#[test]
fn shadowing_shows_single_completion() {
    cov_mark::check!(shadowing_shows_single_completion);

    check_empty(
        r#"
fn foo() {
    let bar = 92;
    {
        let bar = 62;
        drop($0)
    }
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
            lc bar       i32
            fn foo()     fn()
            bt u32
        "#]],
    );
}

#[test]
fn in_macro_expr_frag() {
    check_empty(
        r#"
macro_rules! m { ($e:expr) => { $e } }
fn quux(x: i32) {
    m!($0);
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
            bt u32
            lc x         i32
            fn quux(…)   fn(i32)
            ma m!(…)     macro_rules! m
        "#]],
    );
    check_empty(
        r"
macro_rules! m { ($e:expr) => { $e } }
fn quux(x: i32) {
    m!(x$0);
}
",
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
            bt u32
            lc x         i32
            fn quux(…)   fn(i32)
            ma m!(…)     macro_rules! m
        "#]],
    );
    check_empty(
        r#"
macro_rules! m { ($e:expr) => { $e } }
fn quux(x: i32) {
    let y = 92;
    m!(x$0
}
"#,
        expect![[r#""#]],
    );
}

#[test]
fn enum_qualified() {
    check(
        r#"
impl Enum {
    type AssocType = ();
    const ASSOC_CONST: () = ();
    fn assoc_fn() {}
}
fn func() {
    Enum::$0
}
"#,
        expect![[r#"
            ev TupleV(…)   (u32)
            ev RecordV     {field: u32}
            ev UnitV       ()
            ct ASSOC_CONST const ASSOC_CONST: ()
            fn assoc_fn()  fn()
            ta AssocType   type AssocType = ()
        "#]],
    );
}
