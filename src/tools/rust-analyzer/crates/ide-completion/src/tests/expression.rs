//! Completion tests for expressions.
use expect_test::{Expect, expect};

use crate::{
    CompletionConfig,
    config::AutoImportExclusionType,
    tests::{
        BASE_ITEMS_FIXTURE, TEST_CONFIG, check, check_edit, check_with_base_items,
        completion_list_with_config,
    },
};

fn check_with_config(
    config: CompletionConfig<'_>,
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    expect: Expect,
) {
    let actual = completion_list_with_config(
        config,
        &format!("{BASE_ITEMS_FIXTURE}{ra_fixture}"),
        true,
        None,
    );
    expect.assert_eq(&actual)
}

#[test]
fn complete_literal_struct_with_a_private_field() {
    // `FooDesc.bar` is private, the completion should not be triggered.
    check_with_base_items(
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
            ct CONST                   Unit
            en Enum                    Enum
            fn baz()                   fn()
            fn create_foo(…)   fn(&FooDesc)
            fn function()              fn()
            ma makro!(…) macro_rules! makro
            md _69latrick
            md module
            sc STATIC                  Unit
            st FooDesc              FooDesc
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            un Union                  Union
            ev TupleV(…)        TupleV(u32)
            bt u32                      u32
            kw const
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw mut
            kw raw
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
    check(
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
            fn func(…) fn((i32, i32))
            lc ifletlocal         i32
            lc letlocal           i32
            lc matcharm           i32
            lc param0      (i32, i32)
            lc param1             i32
            lc param2             i32
            bt u32                u32
            kw const
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
            ex ifletlocal
            ex letlocal
            ex matcharm
            ex param1
            ex param2
        "#]],
    );
}

#[test]
fn completes_all_the_things_in_fn_body() {
    check_with_base_items(
        r#"
use non_existent::Unresolved;
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
            ct CONST                   Unit
            cp CONST_PARAM
            en Enum                    Enum
            fn function()              fn()
            fn local_func()            fn()
            me self.foo()          fn(self)
            lc self                    Unit
            ma makro!(…) macro_rules! makro
            md module
            md qualified
            sp Self                    Unit
            sc STATIC                  Unit
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tp TypeParam
            un Union                  Union
            ev TupleV(…)        TupleV(u32)
            bt u32                      u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
            ?? Unresolved
        "#]],
    );
    check_with_base_items(
        r#"
use non_existent::Unresolved;
mod qualified { pub enum Enum { Variant } }

impl Unit {
    fn foo<'lifetime, TypeParam, const CONST_PARAM: usize>(self) {
        fn local_func() {}
        self::$0
    }
}
"#,
        expect![[r#"
            ct CONST                   Unit
            en Enum                    Enum
            fn function()              fn()
            ma makro!(…) macro_rules! makro
            md module
            md qualified
            sc STATIC                  Unit
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            ev TupleV(…)        TupleV(u32)
            ?? Unresolved
        "#]],
    );
}

#[test]
fn complete_in_block() {
    check(
        r#"
    fn foo() {
        if true {
            $0
        }
    }
"#,
        expect![[r#"
            fn foo()  fn()
            bt u32     u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
            ex false
            ex true
        "#]],
    )
}

#[test]
fn complete_after_if_expr() {
    check(
        r#"
    fn foo() {
        if true {}
        $0
    }
"#,
        expect![[r#"
            fn foo()  fn()
            bt u32     u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
    check(
        r#"
    fn foo() {
        match () {
            () => $0
        }
    }
"#,
        expect![[r#"
            fn foo() fn()
            bt u32    u32
            kw const
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
    check(
        r"fn my() { loop { $0 } }",
        expect![[r#"
            fn my()   fn()
            bt u32     u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
    check(
        r"fn my() { loop { foo.$0 } }",
        expect![[r#"
            sn box  Box::new(expr)
            sn break    break expr
            sn call function(expr)
            sn const      const {}
            sn dbg      dbg!(expr)
            sn dbgr    dbg!(&expr)
            sn deref         *expr
            sn if       if expr {}
            sn let             let
            sn letm        let mut
            sn match match expr {}
            sn not           !expr
            sn ref           &expr
            sn refm      &mut expr
            sn return  return expr
            sn unsafe    unsafe {}
            sn while while expr {}
        "#]],
    );
}

#[test]
fn completes_in_let_initializer() {
    check(
        r#"fn main() { let _ = $0 }"#,
        expect![[r#"
            fn main() fn()
            bt u32     u32
            kw const
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
fn completes_after_ref_expr() {
    check(
        r#"fn main() { let _ = &$0 }"#,
        expect![[r#"
            fn main() fn()
            bt u32     u32
            kw const
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw mut
            kw raw
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
        "#]],
    );
    check(
        r#"fn main() { let _ = &raw $0 }"#,
        expect![[r#"
            fn main() fn()
            bt u32     u32
            kw const
            kw const
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
    );
    check(
        r#"fn main() { let _ = &raw const $0 }"#,
        expect![[r#"
            fn main() fn()
            bt u32     u32
            kw const
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
    check(
        r#"fn main() { let _ = &raw mut $0 }"#,
        expect![[r#"
            fn main() fn()
            bt u32     u32
            kw const
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
    check(
        r#"fn main() { let _ = &mut $0 }"#,
        expect![[r#"
            fn main() fn()
            bt u32     u32
            kw const
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
    check(
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
            fn foo() fn()
            st Foo    Foo
            bt u32    u32
            kw const
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

    check(
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
            fn foo() fn()
            lc bar    i32
            bt u32    u32
            kw const
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
    check(
        r#"
macro_rules! m { ($e:expr) => { $e } }
fn quux(x: i32) {
    m!($0);
}
"#,
        expect![[r#"
            fn quux(…)      fn(i32)
            lc x                i32
            ma m!(…) macro_rules! m
            bt u32              u32
            kw const
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
    check(
        r"
macro_rules! m { ($e:expr) => { $e } }
fn quux(x: i32) {
    m!(x$0);
}
",
        expect![[r#"
            fn quux(…)      fn(i32)
            lc x                i32
            ma m!(…) macro_rules! m
            bt u32              u32
            kw const
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
    check(
        r#"
macro_rules! m { ($e:expr) => { $e } }
fn quux(x: i32) {
    let y = 92;
    m!(x$0
}
"#,
        expect![[r#"
            fn quux(…)      fn(i32)
            lc x                i32
            lc y                i32
            ma m!(…) macro_rules! m
            bt u32              u32
            kw const
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
fn enum_qualified() {
    check_with_base_items(
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
            ct ASSOC_CONST  const ASSOC_CONST: ()
            fn assoc_fn()                    fn()
            ta AssocType      type AssocType = ()
            ev RecordV {…} RecordV { field: u32 }
            ev TupleV(…)              TupleV(u32)
            ev UnitV                        UnitV
        "#]],
    );
}

#[test]
fn ty_qualified_no_drop() {
    check(
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
    check(
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
            ev Variant      Variant
        "#]],
    );
}

#[test]
fn detail_impl_trait_in_return_position() {
    check(
        r"
//- minicore: sized
trait Trait<T> {}
fn foo<U>() -> impl Trait<U> {}
fn main() {
    self::$0
}
",
        expect![[r#"
            fn foo() fn() -> impl Trait<U>
            fn main()                 fn()
            tt Trait
        "#]],
    );
}

#[test]
fn detail_async_fn() {
    check(
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
            fn bar() async fn() -> impl Trait<U>
            fn foo()            async fn() -> u8
            fn main()                       fn()
            tt Trait
        "#]],
    );
}

#[test]
fn detail_impl_trait_in_argument_position() {
    check(
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
        expect![[r#"
            fn bar(…) fn(impl Trait<U>)
        "#]],
    );
}

#[test]
fn complete_record_expr_path() {
    check_with_base_items(
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
            ex Zulu
            ex Zulu::test()
        "#]],
    );
}

#[test]
fn variant_with_struct() {
    check(
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
            en HH                                    HH
            fn brr()                               fn()
            st YoloVariant                  YoloVariant
            st YoloVariant {…} YoloVariant { f: usize }
            bt u32                                  u32
            kw const
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
    check(
        r#"
fn foo() { if foo {} $0 }
"#,
        expect![[r#"
            fn foo()  fn()
            bt u32     u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
    check(
        r#"
fn foo() { if foo {} el$0 }
"#,
        expect![[r#"
            fn foo()  fn()
            bt u32     u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
    check(
        r#"
fn foo() { bar(if foo {} $0) }
"#,
        expect![[r#"
            fn foo() fn()
            bt u32    u32
            kw const
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
    check(
        r#"
fn foo() { bar(if foo {} el$0) }
"#,
        expect![[r#"
            fn foo() fn()
            bt u32    u32
            kw const
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
    check(
        r#"
fn foo() { if foo {} $0 let x = 92; }
"#,
        expect![[r#"
            fn foo()  fn()
            bt u32     u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
    check(
        r#"
fn foo() { if foo {} el$0 let x = 92; }
"#,
        expect![[r#"
            fn foo()  fn()
            bt u32     u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
    check(
        r#"
fn foo() { if foo {} el$0 { let x = 92; } }
"#,
        expect![[r#"
            fn foo()  fn()
            bt u32     u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
    check(
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
            fn main() fn()
            md std
            bt u32     u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
    check(
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
            fn main()                                                     fn()
            md std
            st UnstableButWeAreOnNightlyAnyway UnstableButWeAreOnNightlyAnyway
            bt u32                                                         u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
fn inside_format_args_completions_work() {
    check(
        r#"
//- minicore: fmt
struct Foo;
impl Foo {
    fn foo(&self) {}
}

fn main() {
    format_args!("{}", Foo.$0);
}
"#,
        expect![[r#"
            me foo()     fn(&self)
            sn box  Box::new(expr)
            sn call function(expr)
            sn const      const {}
            sn dbg      dbg!(expr)
            sn dbgr    dbg!(&expr)
            sn deref         *expr
            sn match match expr {}
            sn ref           &expr
            sn refm      &mut expr
            sn return  return expr
            sn unsafe    unsafe {}
        "#]],
    );
    check(
        r#"
//- minicore: fmt
struct Foo;
impl Foo {
    fn foo(&self) {}
}

fn main() {
    format_args!("{}", Foo.f$0);
}
"#,
        expect![[r#"
            me foo()     fn(&self)
            sn box  Box::new(expr)
            sn call function(expr)
            sn const      const {}
            sn dbg      dbg!(expr)
            sn dbgr    dbg!(&expr)
            sn deref         *expr
            sn match match expr {}
            sn ref           &expr
            sn refm      &mut expr
            sn return  return expr
            sn unsafe    unsafe {}
        "#]],
    );
}

#[test]
fn inside_faulty_format_args_completions_work() {
    check(
        r#"
//- minicore: fmt
struct Foo;
impl Foo {
    fn foo(&self) {}
}

fn main() {
    format_args!("", Foo.$0);
}
"#,
        expect![[r#"
            me foo()     fn(&self)
            sn box  Box::new(expr)
            sn call function(expr)
            sn const      const {}
            sn dbg      dbg!(expr)
            sn dbgr    dbg!(&expr)
            sn deref         *expr
            sn match match expr {}
            sn ref           &expr
            sn refm      &mut expr
            sn return  return expr
            sn unsafe    unsafe {}
        "#]],
    );
    check(
        r#"
//- minicore: fmt
struct Foo;
impl Foo {
    fn foo(&self) {}
}

fn main() {
    format_args!("", Foo.f$0);
}
"#,
        expect![[r#"
            me foo()     fn(&self)
            sn box  Box::new(expr)
            sn call function(expr)
            sn const      const {}
            sn dbg      dbg!(expr)
            sn dbgr    dbg!(&expr)
            sn deref         *expr
            sn match match expr {}
            sn ref           &expr
            sn refm      &mut expr
            sn return  return expr
            sn unsafe    unsafe {}
        "#]],
    );
    check(
        r#"
//- minicore: fmt
struct Foo;
impl Foo {
    fn foo(&self) {}
}

fn main() {
    format_args!("{} {named} {captured} {named} {}", a, named = c, Foo.f$0);
}
"#,
        expect![[r#"
            me foo()     fn(&self)
            sn box  Box::new(expr)
            sn call function(expr)
            sn const      const {}
            sn dbg      dbg!(expr)
            sn dbgr    dbg!(&expr)
            sn deref         *expr
            sn match match expr {}
            sn ref           &expr
            sn refm      &mut expr
            sn return  return expr
            sn unsafe    unsafe {}
        "#]],
    );
    check(
        r#"
//- minicore: fmt
struct Foo;
impl Foo {
    fn foo(&self) {}
}

fn main() {
    format_args!("{", Foo.f$0);
}
"#,
        expect![[r#"
            me foo()     fn(&self)
            sn box  Box::new(expr)
            sn call function(expr)
            sn const      const {}
            sn dbg      dbg!(expr)
            sn dbgr    dbg!(&expr)
            sn deref         *expr
            sn match match expr {}
            sn ref           &expr
            sn refm      &mut expr
            sn return  return expr
            sn unsafe    unsafe {}
        "#]],
    );
}

#[test]
fn macro_that_ignores_completion_marker() {
    check_with_base_items(
        r#"
macro_rules! helper {
    ($v:ident) => {};
}

macro_rules! m {
    ($v:ident) => {{
        helper!($v);
        $v
    }};
}

fn main() {
    let variable = "test";
    m!(v$0);
}
    "#,
        expect![[r#"
            ct CONST                     Unit
            en Enum                      Enum
            fn function()                fn()
            fn main()                    fn()
            lc variable          &'static str
            ma helper!(…) macro_rules! helper
            ma m!(…)           macro_rules! m
            ma makro!(…)   macro_rules! makro
            md module
            sc STATIC                    Unit
            st Record                  Record
            st Tuple                    Tuple
            st Unit                      Unit
            un Union                    Union
            ev TupleV(…)          TupleV(u32)
            bt u32                        u32
            kw async
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
            kw impl for
            kw let
            kw letm
            kw loop
            kw match
            kw mod
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
fn excluded_trait_method_is_excluded() {
    check_with_config(
        CompletionConfig {
            exclude_traits: &["ra_test_fixture::ExcludedTrait".to_owned()],
            ..TEST_CONFIG
        },
        r#"
trait ExcludedTrait {
    fn foo(&self) {}
    fn bar(&self) {}
    fn baz(&self) {}
}

impl<T> ExcludedTrait for T {}

struct Foo;
impl Foo {
    fn inherent(&self) {}
}

fn foo() {
    Foo.$0
}
        "#,
        expect![[r#"
            me inherent() fn(&self)
            sn box   Box::new(expr)
            sn call  function(expr)
            sn const       const {}
            sn dbg       dbg!(expr)
            sn dbgr     dbg!(&expr)
            sn deref          *expr
            sn let              let
            sn letm         let mut
            sn match  match expr {}
            sn ref            &expr
            sn refm       &mut expr
            sn return   return expr
            sn unsafe     unsafe {}
        "#]],
    );
}

#[test]
fn excluded_trait_not_excluded_when_inherent() {
    check_with_config(
        CompletionConfig {
            exclude_traits: &["ra_test_fixture::ExcludedTrait".to_owned()],
            ..TEST_CONFIG
        },
        r#"
trait ExcludedTrait {
    fn foo(&self) {}
    fn bar(&self) {}
    fn baz(&self) {}
}

impl<T> ExcludedTrait for T {}

fn foo(v: &dyn ExcludedTrait) {
    v.$0
}
        "#,
        expect![[r#"
            me bar() (as ExcludedTrait) fn(&self)
            me baz() (as ExcludedTrait) fn(&self)
            me foo() (as ExcludedTrait) fn(&self)
            sn box                 Box::new(expr)
            sn call                function(expr)
            sn const                     const {}
            sn dbg                     dbg!(expr)
            sn dbgr                   dbg!(&expr)
            sn deref                        *expr
            sn let                            let
            sn letm                       let mut
            sn match                match expr {}
            sn ref                          &expr
            sn refm                     &mut expr
            sn return                 return expr
            sn unsafe                   unsafe {}
        "#]],
    );
    check_with_config(
        CompletionConfig {
            exclude_traits: &["ra_test_fixture::ExcludedTrait".to_owned()],
            ..TEST_CONFIG
        },
        r#"
trait ExcludedTrait {
    fn foo(&self) {}
    fn bar(&self) {}
    fn baz(&self) {}
}

impl<T> ExcludedTrait for T {}

fn foo(v: impl ExcludedTrait) {
    v.$0
}
        "#,
        expect![[r#"
            me bar() (as ExcludedTrait) fn(&self)
            me baz() (as ExcludedTrait) fn(&self)
            me foo() (as ExcludedTrait) fn(&self)
            sn box                 Box::new(expr)
            sn call                function(expr)
            sn const                     const {}
            sn dbg                     dbg!(expr)
            sn dbgr                   dbg!(&expr)
            sn deref                        *expr
            sn let                            let
            sn letm                       let mut
            sn match                match expr {}
            sn ref                          &expr
            sn refm                     &mut expr
            sn return                 return expr
            sn unsafe                   unsafe {}
        "#]],
    );
    check_with_config(
        CompletionConfig {
            exclude_traits: &["ra_test_fixture::ExcludedTrait".to_owned()],
            ..TEST_CONFIG
        },
        r#"
trait ExcludedTrait {
    fn foo(&self) {}
    fn bar(&self) {}
    fn baz(&self) {}
}

impl<T> ExcludedTrait for T {}

fn foo<T: ExcludedTrait>(v: T) {
    v.$0
}
        "#,
        expect![[r#"
            me bar() (as ExcludedTrait) fn(&self)
            me baz() (as ExcludedTrait) fn(&self)
            me foo() (as ExcludedTrait) fn(&self)
            sn box                 Box::new(expr)
            sn call                function(expr)
            sn const                     const {}
            sn dbg                     dbg!(expr)
            sn dbgr                   dbg!(&expr)
            sn deref                        *expr
            sn let                            let
            sn letm                       let mut
            sn match                match expr {}
            sn ref                          &expr
            sn refm                     &mut expr
            sn return                 return expr
            sn unsafe                   unsafe {}
        "#]],
    );
}

#[test]
fn excluded_trait_method_is_excluded_from_flyimport() {
    check_with_config(
        CompletionConfig {
            exclude_traits: &["ra_test_fixture::module2::ExcludedTrait".to_owned()],
            ..TEST_CONFIG
        },
        r#"
mod module2 {
    pub trait ExcludedTrait {
        fn foo(&self) {}
        fn bar(&self) {}
        fn baz(&self) {}
    }

    impl<T> ExcludedTrait for T {}
}

struct Foo;
impl Foo {
    fn inherent(&self) {}
}

fn foo() {
    Foo.$0
}
        "#,
        expect![[r#"
            me inherent() fn(&self)
            sn box   Box::new(expr)
            sn call  function(expr)
            sn const       const {}
            sn dbg       dbg!(expr)
            sn dbgr     dbg!(&expr)
            sn deref          *expr
            sn let              let
            sn letm         let mut
            sn match  match expr {}
            sn ref            &expr
            sn refm       &mut expr
            sn return   return expr
            sn unsafe     unsafe {}
        "#]],
    );
}

#[test]
fn flyimport_excluded_trait_method_is_excluded_from_flyimport() {
    check_with_config(
        CompletionConfig {
            exclude_flyimport: vec![(
                "ra_test_fixture::module2::ExcludedTrait".to_owned(),
                AutoImportExclusionType::Methods,
            )],
            ..TEST_CONFIG
        },
        r#"
mod module2 {
    pub trait ExcludedTrait {
        fn foo(&self) {}
        fn bar(&self) {}
        fn baz(&self) {}
    }

    impl<T> ExcludedTrait for T {}
}

struct Foo;
impl Foo {
    fn inherent(&self) {}
}

fn foo() {
    Foo.$0
}
        "#,
        expect![[r#"
            me inherent() fn(&self)
            sn box   Box::new(expr)
            sn call  function(expr)
            sn const       const {}
            sn dbg       dbg!(expr)
            sn dbgr     dbg!(&expr)
            sn deref          *expr
            sn let              let
            sn letm         let mut
            sn match  match expr {}
            sn ref            &expr
            sn refm       &mut expr
            sn return   return expr
            sn unsafe     unsafe {}
        "#]],
    );
}

#[test]
fn excluded_trait_method_is_excluded_from_path_completion() {
    check_with_config(
        CompletionConfig {
            exclude_traits: &["ra_test_fixture::ExcludedTrait".to_owned()],
            ..TEST_CONFIG
        },
        r#"
pub trait ExcludedTrait {
    fn foo(&self) {}
    fn bar(&self) {}
    fn baz(&self) {}
}

impl<T> ExcludedTrait for T {}

struct Foo;
impl Foo {
    fn inherent(&self) {}
}

fn foo() {
    Foo::$0
}
        "#,
        expect![[r#"
            me inherent(…) fn(&self)
        "#]],
    );
}

#[test]
fn excluded_trait_method_is_not_excluded_when_trait_is_specified() {
    check_with_config(
        CompletionConfig {
            exclude_traits: &["ra_test_fixture::ExcludedTrait".to_owned()],
            ..TEST_CONFIG
        },
        r#"
pub trait ExcludedTrait {
    fn foo(&self) {}
    fn bar(&self) {}
    fn baz(&self) {}
}

impl<T> ExcludedTrait for T {}

struct Foo;
impl Foo {
    fn inherent(&self) {}
}

fn foo() {
    ExcludedTrait::$0
}
        "#,
        expect![[r#"
                me bar(…) (as ExcludedTrait) fn(&self)
                me baz(…) (as ExcludedTrait) fn(&self)
                me foo(…) (as ExcludedTrait) fn(&self)
            "#]],
    );
    check_with_config(
        CompletionConfig {
            exclude_traits: &["ra_test_fixture::ExcludedTrait".to_owned()],
            ..TEST_CONFIG
        },
        r#"
pub trait ExcludedTrait {
    fn foo(&self) {}
    fn bar(&self) {}
    fn baz(&self) {}
}

impl<T> ExcludedTrait for T {}

struct Foo;
impl Foo {
    fn inherent(&self) {}
}

fn foo() {
    <Foo as ExcludedTrait>::$0
}
        "#,
        expect![[r#"
                me bar(…) (as ExcludedTrait) fn(&self)
                me baz(…) (as ExcludedTrait) fn(&self)
                me foo(…) (as ExcludedTrait) fn(&self)
            "#]],
    );
}

#[test]
fn excluded_trait_not_excluded_when_inherent_path() {
    check_with_config(
        CompletionConfig {
            exclude_traits: &["ra_test_fixture::ExcludedTrait".to_owned()],
            ..TEST_CONFIG
        },
        r#"
trait ExcludedTrait {
    fn foo(&self) {}
    fn bar(&self) {}
    fn baz(&self) {}
}

impl<T> ExcludedTrait for T {}

fn foo() {
    <dyn ExcludedTrait>::$0
}
        "#,
        expect![[r#"
            me bar(…) (as ExcludedTrait) fn(&self)
            me baz(…) (as ExcludedTrait) fn(&self)
            me foo(…) (as ExcludedTrait) fn(&self)
        "#]],
    );
    check_with_config(
        CompletionConfig {
            exclude_traits: &["ra_test_fixture::ExcludedTrait".to_owned()],
            ..TEST_CONFIG
        },
        r#"
trait ExcludedTrait {
    fn foo(&self) {}
    fn bar(&self) {}
    fn baz(&self) {}
}

impl<T> ExcludedTrait for T {}

fn foo<T: ExcludedTrait>() {
    T::$0
}
        "#,
        expect![[r#"
            me bar(…) (as ExcludedTrait) fn(&self)
            me baz(…) (as ExcludedTrait) fn(&self)
            me foo(…) (as ExcludedTrait) fn(&self)
        "#]],
    );
}

#[test]
fn hide_ragennew_synthetic_identifiers() {
    check(
        r#"
//- minicore: iterator
fn bar() {
    for i in [0; 10] {
        r$0
    }
}
        "#,
        expect![[r#"
            en Option                             Option<{unknown}>
            en Result                  Result<{unknown}, {unknown}>
            fn bar()                                           fn()
            lc i                                                i32
            ma const_format_args!(…) macro_rules! const_format_args
            ma format_args!(…)             macro_rules! format_args
            ma format_args_nl!(…)       macro_rules! format_args_nl
            ma panic!(…)                         macro_rules! panic
            ma print!(…)                         macro_rules! print
            md core
            md result (use core::result)
            md rust_2015 (use core::prelude::rust_2015)
            md rust_2018 (use core::prelude::rust_2018)
            md rust_2021 (use core::prelude::rust_2021)
            md rust_2024 (use core::prelude::rust_2024)
            tt Clone
            tt Copy
            tt IntoIterator
            tt Iterator
            ta Result (use core::fmt::Result)
            ev Err(…)                                        Err(E)
            ev None                                            None
            ev Ok(…)                                          Ok(T)
            ev Some(…)                                      Some(T)
            bt u32                                              u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
fn doc_hidden_enum_variant() {
    check(
        r#"
//- /foo.rs crate:foo
pub enum Enum {
    #[doc(hidden)] Hidden,
    Visible,
}

//- /lib.rs crate:lib deps:foo
fn foo() {
    let _ = foo::Enum::$0;
}
    "#,
        expect![[r#"
            ev Visible Visible
        "#]],
    );
}

#[test]
fn non_std_test_attr_macro() {
    check(
        r#"
//- proc_macros: identity
use proc_macros::identity as test;

#[test]
fn foo() {
    $0
}
    "#,
        expect![[r#"
            fn foo()  fn()
            md proc_macros
            bt u32     u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
fn cfg_attr_attr_macro() {
    check(
        r#"
//- proc_macros: identity
#[cfg_attr(test, proc_macros::identity)]
fn foo() {
    $0
}
    "#,
        expect![[r#"
            fn foo()  fn()
            md proc_macros
            bt u32     u32
            kw async
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
            kw impl for
            kw let
            kw letm
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
fn escaped_label() {
    check(
        r#"
fn main() {
    'r#break: {
        break '$0;
    }
}
    "#,
        expect![[r#"
            lb 'r#break
        "#]],
    );
}

#[test]
fn call_parens_with_newline() {
    check_edit(
        "foo",
        r#"
fn foo(v: i32) {}

fn bar() {
    foo$0
    ()
}
    "#,
        r#"
fn foo(v: i32) {}

fn bar() {
    foo(${1:v});$0
    ()
}
    "#,
    );
    check_edit(
        "foo",
        r#"
struct Foo;
impl Foo {
    fn foo(&self, v: i32) {}
}

fn bar() {
    Foo.foo$0
    ()
}
    "#,
        r#"
struct Foo;
impl Foo {
    fn foo(&self, v: i32) {}
}

fn bar() {
    Foo.foo(${1:v});$0
    ()
}
    "#,
    );
}

#[test]
fn dbg_too_many_asterisks() {
    check_edit(
        "dbg",
        r#"
fn main() {
    let x = &42;
    let y = *x.$0;
}
    "#,
        r#"
fn main() {
    let x = &42;
    let y = dbg!(*x);
}
    "#,
    );
}
