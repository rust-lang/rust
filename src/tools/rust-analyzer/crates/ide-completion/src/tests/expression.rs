//! Completion tests for expressions.
use expect_test::{expect, Expect};

use crate::tests::{check_edit, check_empty, completion_list, BASE_ITEMS_FIXTURE};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(&format!("{BASE_ITEMS_FIXTURE}{ra_fixture}"));
    expect.assert_eq(&actual)
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
        expect![[r#"
            ct CONST
            en Enum
            fn baz()         fn()
            fn create_foo(…) fn(&FooDesc)
            fn function()    fn()
            ma makro!(…)     macro_rules! makro
            md _69latrick
            md module
            sc STATIC
            st FooDesc
            st Record
            st Tuple
            st Unit
            un Union
            ev TupleV(…)     TupleV(u32)
            bt u32
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw mut
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
        "#]],
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
            fn func(…)    fn((i32, i32))
            lc ifletlocal i32
            lc letlocal   i32
            lc matcharm   i32
            lc param0     (i32, i32)
            lc param1     i32
            lc param2     i32
            bt u32
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
        "#]],
    );
}

#[test]
fn completes_all_the_things_in_fn_body() {
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
        expect![[r#"
            ct CONST
            cp CONST_PARAM
            en Enum
            fn function()   fn()
            fn local_func() fn()
            lc self         Unit
            ma makro!(…)    macro_rules! makro
            md module
            md qualified
            sp Self
            sc STATIC
            st Record
            st Tuple
            st Unit
            tp TypeParam
            un Union
            ev TupleV(…)    TupleV(u32)
            bt u32
            kw const
            kw crate::
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            me self.foo()   fn(self)
            sn macro_rules
            sn pd
            sn ppd
            ?? Unresolved
        "#]],
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
        expect![[r#"
            ct CONST
            en Enum
            fn function() fn()
            ma makro!(…)  macro_rules! makro
            md module
            md qualified
            sc STATIC
            st Record
            st Tuple
            st Unit
            tt Trait
            un Union
            ev TupleV(…)  TupleV(u32)
            ?? Unresolved
        "#]],
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
            fn foo()       fn()
            bt u32
            kw const
            kw crate::
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            sn macro_rules
            sn pd
            sn ppd
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
            fn foo()       fn()
            bt u32
            kw const
            kw crate::
            kw else
            kw else if
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            sn macro_rules
            sn pd
            sn ppd
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
            fn foo()     fn()
            bt u32
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
        "#]],
    )
}

#[test]
fn completes_in_loop_ctx() {
    check_empty(
        r"fn my() { loop { $0 } }",
        expect![[r#"
            fn my()        fn()
            bt u32
            kw break
            kw const
            kw continue
            kw crate::
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            sn macro_rules
            sn pd
            sn ppd
        "#]],
    );
}

#[test]
fn completes_in_let_initializer() {
    check_empty(
        r#"fn main() { let _ = $0 }"#,
        expect![[r#"
            fn main()    fn()
            bt u32
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
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
            fn foo()     fn()
            st Foo
            bt u32
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
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
            fn foo()     fn()
            lc bar       i32
            bt u32
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
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
            fn quux(…)   fn(i32)
            lc x         i32
            ma m!(…)     macro_rules! m
            bt u32
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
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
            fn quux(…)   fn(i32)
            lc x         i32
            ma m!(…)     macro_rules! m
            bt u32
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
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
            ct ASSOC_CONST const ASSOC_CONST: ()
            fn assoc_fn()  fn()
            ta AssocType   type AssocType = ()
            ev RecordV {…} RecordV { field: u32 }
            ev TupleV(…)   TupleV(u32)
            ev UnitV       UnitV
        "#]],
    );
}

#[test]
fn ty_qualified_no_drop() {
    check_empty(
        r#"
//- minicore: drop
struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {}
}
fn func() {
    Foo::$0
}
"#,
        expect![[r#""#]],
    );
}

#[test]
fn with_parens() {
    check_empty(
        r#"
enum Enum {
    Variant()
}
impl Enum {
    fn variant() -> Self { Enum::Variant() }
}
fn func() {
    Enum::$0()
}
"#,
        expect![[r#"
            fn variant fn() -> Enum
            ev Variant Variant
        "#]],
    );
}

#[test]
fn detail_impl_trait_in_return_position() {
    check_empty(
        r"
//- minicore: sized
trait Trait<T> {}
fn foo<U>() -> impl Trait<U> {}
fn main() {
    self::$0
}
",
        expect![[r#"
            fn foo()  fn() -> impl Trait<U>
            fn main() fn()
            tt Trait
        "#]],
    );
}

#[test]
fn detail_async_fn() {
    check_empty(
        r#"
//- minicore: future, sized
trait Trait<T> {}
async fn foo() -> u8 {}
async fn bar<U>() -> impl Trait<U> {}
fn main() {
    self::$0
}
"#,
        expect![[r#"
            fn bar()  async fn() -> impl Trait<U>
            fn foo()  async fn() -> u8
            fn main() fn()
            tt Trait
        "#]],
    );
}

#[test]
fn detail_impl_trait_in_argument_position() {
    check_empty(
        r"
//- minicore: sized
trait Trait<T> {}
struct Foo;
impl Foo {
    fn bar<U>(_: impl Trait<U>) {}
}
fn main() {
    Foo::$0
}
",
        expect![[r"
            fn bar(…) fn(impl Trait<U>)
        "]],
    );
}

#[test]
fn complete_record_expr_path() {
    check(
        r#"
struct Zulu;
impl Zulu {
    fn test() -> Self { }
}
fn boi(val: Zulu) { }
fn main() {
    boi(Zulu:: $0 {});
}
"#,
        expect![[r#"
            fn test() fn() -> Zulu
        "#]],
    );
}

#[test]
fn variant_with_struct() {
    check_empty(
        r#"
pub struct YoloVariant {
    pub f: usize
}

pub enum HH {
    Yolo(YoloVariant),
}

fn brr() {
    let t = HH::Yolo(Y$0);
}
"#,
        expect![[r#"
            en HH
            fn brr()           fn()
            st YoloVariant
            st YoloVariant {…} YoloVariant { f: usize }
            bt u32
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
        "#]],
    );
}

#[test]
fn return_unit_block() {
    cov_mark::check!(return_unit_block);
    check_edit("return", r#"fn f() { if true { $0 } }"#, r#"fn f() { if true { return; } }"#);
}

#[test]
fn return_unit_no_block() {
    cov_mark::check!(return_unit_no_block);
    check_edit(
        "return",
        r#"fn f() { match () { () => $0 } }"#,
        r#"fn f() { match () { () => return } }"#,
    );
}

#[test]
fn return_value_block() {
    cov_mark::check!(return_value_block);
    check_edit(
        "return",
        r#"fn f() -> i32 { if true { $0 } }"#,
        r#"fn f() -> i32 { if true { return $0; } }"#,
    );
}

#[test]
fn return_value_no_block() {
    cov_mark::check!(return_value_no_block);
    check_edit(
        "return",
        r#"fn f() -> i32 { match () { () => $0 } }"#,
        r#"fn f() -> i32 { match () { () => return $0 } }"#,
    );
}

#[test]
fn else_completion_after_if() {
    check_empty(
        r#"
fn foo() { if foo {} $0 }
"#,
        expect![[r#"
            fn foo()       fn()
            bt u32
            kw const
            kw crate::
            kw else
            kw else if
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            sn macro_rules
            sn pd
            sn ppd
        "#]],
    );
    check_empty(
        r#"
fn foo() { if foo {} el$0 }
"#,
        expect![[r#"
            fn foo()       fn()
            bt u32
            kw const
            kw crate::
            kw else
            kw else if
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            sn macro_rules
            sn pd
            sn ppd
        "#]],
    );
    check_empty(
        r#"
fn foo() { bar(if foo {} $0) }
"#,
        expect![[r#"
            fn foo()     fn()
            bt u32
            kw crate::
            kw else
            kw else if
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
        "#]],
    );
    check_empty(
        r#"
fn foo() { bar(if foo {} el$0) }
"#,
        expect![[r#"
            fn foo()     fn()
            bt u32
            kw crate::
            kw else
            kw else if
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
        "#]],
    );
    check_empty(
        r#"
fn foo() { if foo {} $0 let x = 92; }
"#,
        expect![[r#"
            fn foo()       fn()
            bt u32
            kw const
            kw crate::
            kw else
            kw else if
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            sn macro_rules
            sn pd
            sn ppd
        "#]],
    );
    check_empty(
        r#"
fn foo() { if foo {} el$0 let x = 92; }
"#,
        expect![[r#"
            fn foo()       fn()
            bt u32
            kw const
            kw crate::
            kw else
            kw else if
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            sn macro_rules
            sn pd
            sn ppd
        "#]],
    );
    check_empty(
        r#"
fn foo() { if foo {} el$0 { let x = 92; } }
"#,
        expect![[r#"
            fn foo()       fn()
            bt u32
            kw const
            kw crate::
            kw else
            kw else if
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            sn macro_rules
            sn pd
            sn ppd
        "#]],
    );
}

#[test]
fn expr_no_unstable_item_on_stable() {
    check_empty(
        r#"
//- /main.rs crate:main deps:std
use std::*;
fn main() {
    $0
}
//- /std.rs crate:std
#[unstable]
pub struct UnstableThisShouldNotBeListed;
"#,
        expect![[r#"
            fn main()      fn()
            md std
            bt u32
            kw const
            kw crate::
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            sn macro_rules
            sn pd
            sn ppd
        "#]],
    );
}

#[test]
fn expr_unstable_item_on_nightly() {
    check_empty(
        r#"
//- toolchain:nightly
//- /main.rs crate:main deps:std
use std::*;
fn main() {
    $0
}
//- /std.rs crate:std
#[unstable]
pub struct UnstableButWeAreOnNightlyAnyway;
"#,
        expect![[r#"
            fn main()                 fn()
            md std
            st UnstableButWeAreOnNightlyAnyway
            bt u32
            kw const
            kw crate::
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            sn macro_rules
            sn pd
            sn ppd
        "#]],
    );
}
