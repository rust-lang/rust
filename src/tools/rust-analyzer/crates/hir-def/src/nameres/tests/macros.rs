use expect_test::expect;

use itertools::Itertools;
use span::Edition;

use super::*;

#[test]
fn macro_rules_are_globally_visible() {
    check(
        r#"
//- /lib.rs
macro_rules! structs {
    ($($i:ident),*) => {
        $(struct $i { field: u32 } )*
    }
}
structs!(Foo);
mod nested;

//- /nested.rs
structs!(Bar, Baz);
"#,
        expect![[r#"
            crate
            Foo: t
            nested: t

            crate::nested
            Bar: t
            Baz: t
        "#]],
    );
}

#[test]
fn macro_rules_can_define_modules() {
    check(
        r#"
//- /lib.rs
macro_rules! m {
    ($name:ident) => { mod $name;  }
}
m!(n1);
mod m { m!(n3) }

//- /n1.rs
m!(n2)
//- /n1/n2.rs
struct X;
//- /m/n3.rs
struct Y;
"#,
        expect![[r#"
            crate
            m: t
            n1: t

            crate::m
            n3: t

            crate::m::n3
            Y: t v

            crate::n1
            n2: t

            crate::n1::n2
            X: t v
        "#]],
    );
}

#[test]
fn macro_rules_from_other_crates_are_visible() {
    check(
        r#"
//- /main.rs crate:main deps:foo
foo::structs!(Foo, Bar)
mod bar;

//- /bar.rs
use crate::*;

//- /lib.rs crate:foo
#[macro_export]
macro_rules! structs {
    ($($i:ident),*) => {
        $(struct $i { field: u32 } )*
    }
}
"#,
        expect![[r#"
            crate
            Bar: t
            Foo: t
            bar: t

            crate::bar
            Bar: tg
            Foo: tg
            bar: tg
        "#]],
    );
}

#[test]
fn macro_rules_export_with_local_inner_macros_are_visible() {
    check(
        r#"
//- /main.rs crate:main deps:foo
foo::structs!(Foo, Bar)
mod bar;

//- /bar.rs
use crate::*;

//- /lib.rs crate:foo
#[macro_export(local_inner_macros)]
macro_rules! structs {
    ($($i:ident),*) => {
        $(struct $i { field: u32 } )*
    }
}
"#,
        expect![[r#"
            crate
            Bar: t
            Foo: t
            bar: t

            crate::bar
            Bar: tg
            Foo: tg
            bar: tg
        "#]],
    );
}

#[test]
fn local_inner_macros_makes_local_macros_usable() {
    check(
        r#"
//- /main.rs crate:main deps:foo
foo::structs!(Foo, Bar);
mod bar;

//- /bar.rs
use crate::*;

//- /lib.rs crate:foo
#[macro_export(local_inner_macros)]
macro_rules! structs {
    ($($i:ident),*) => {
        inner!($($i),*);
    }
}
#[macro_export]
macro_rules! inner {
    ($($i:ident),*) => {
        $(struct $i { field: u32 } )*
    }
}
"#,
        expect![[r#"
            crate
            Bar: t
            Foo: t
            bar: t

            crate::bar
            Bar: tg
            Foo: tg
            bar: tg
        "#]],
    );
}

#[test]
fn unexpanded_macro_should_expand_by_fixedpoint_loop() {
    check(
        r#"
//- /main.rs crate:main deps:foo
macro_rules! baz {
    () => {
        use foo::bar;
    }
}
foo!();
bar!();
baz!();

//- /lib.rs crate:foo
#[macro_export]
macro_rules! foo {
    () => {
        struct Foo { field: u32 }
    }
}
#[macro_export]
macro_rules! bar {
    () => {
        use foo::foo;
    }
}
"#,
        expect![[r#"
            crate
            Foo: t
            bar: mi
            foo: mi
        "#]],
    );
}

#[test]
fn macro_rules_from_other_crates_are_visible_with_macro_use() {
    cov_mark::check!(macro_rules_from_other_crates_are_visible_with_macro_use);
    check(
        r#"
//- /main.rs crate:main deps:foo
structs!(Foo);
structs_priv!(Bar);
structs_not_exported!(MacroNotResolved1);
crate::structs!(MacroNotResolved2);

mod bar;

#[macro_use]
extern crate foo;

//- /bar.rs
structs!(Baz);
crate::structs!(MacroNotResolved3);

//- /lib.rs crate:foo
#[macro_export]
macro_rules! structs {
    ($i:ident) => { struct $i; }
}

macro_rules! structs_not_exported {
    ($i:ident) => { struct $i; }
}

mod priv_mod {
    #[macro_export]
    macro_rules! structs_priv {
        ($i:ident) => { struct $i; }
    }
}
"#,
        expect![[r#"
            crate
            Bar: t v
            Foo: t v
            bar: t
            foo: te

            crate::bar
            Baz: t v
        "#]],
    );
}

#[test]
fn macro_use_filter() {
    check(
        r#"
//- /main.rs crate:main deps:empty,multiple,all
#[macro_use()]
extern crate empty;

foo_not_imported!();

#[macro_use(bar1)]
#[macro_use()]
#[macro_use(bar2, bar3)]
extern crate multiple;

bar1!();
bar2!();
bar3!();
bar_not_imported!();

#[macro_use(baz1)]
#[macro_use]
#[macro_use(baz2)]
extern crate all;

baz1!();
baz2!();
baz3!();

//- /empty.rs crate:empty
#[macro_export]
macro_rules! foo_not_imported { () => { struct NotOkFoo; } }

//- /multiple.rs crate:multiple
#[macro_export]
macro_rules! bar1 { () => { struct OkBar1; } }
#[macro_export]
macro_rules! bar2 { () => { struct OkBar2; } }
#[macro_export]
macro_rules! bar3 { () => { struct OkBar3; } }
#[macro_export]
macro_rules! bar_not_imported { () => { struct NotOkBar; } }

//- /all.rs crate:all
#[macro_export]
macro_rules! baz1 { () => { struct OkBaz1; } }
#[macro_export]
macro_rules! baz2 { () => { struct OkBaz2; } }
#[macro_export]
macro_rules! baz3 { () => { struct OkBaz3; } }
"#,
        expect![[r#"
            crate
            OkBar1: t v
            OkBar2: t v
            OkBar3: t v
            OkBaz1: t v
            OkBaz2: t v
            OkBaz3: t v
            all: te
            empty: te
            multiple: te
        "#]],
    );
}

#[test]
fn prelude_is_macro_use() {
    cov_mark::check!(prelude_is_macro_use);
    check(
        r#"
//- /main.rs edition:2018 crate:main deps:std
structs!(Foo);
structs_priv!(Bar);
structs_outside!(Out);
crate::structs!(MacroNotResolved2);

mod bar;

//- /bar.rs
structs!(Baz);
crate::structs!(MacroNotResolved3);

//- /lib.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        #[macro_export]
        macro_rules! structs {
            ($i:ident) => { struct $i; }
        }

        mod priv_mod {
            #[macro_export]
            macro_rules! structs_priv {
                ($i:ident) => { struct $i; }
            }
        }
    }
}

#[macro_export]
macro_rules! structs_outside {
    ($i:ident) => { struct $i; }
}
"#,
        expect![[r#"
            crate
            Bar: t v
            Foo: t v
            Out: t v
            bar: t

            crate::bar
            Baz: t v
        "#]],
    );
}

#[test]
fn prelude_cycle() {
    check(
        r#"
#[prelude_import]
use self::prelude::*;

declare_mod!();

mod prelude {
    macro_rules! declare_mod {
        () => (mod foo {})
    }
}
"#,
        expect![[r#"
            crate
            prelude: t

            crate::prelude
        "#]],
    );
}

#[test]
fn legacy_macro_use_before_def() {
    check(
        r#"
m!();

macro_rules! m {
    () => {
        struct S;
    }
}
"#,
        expect![[r#"
            crate
            S: t v
        "#]],
    );
    // FIXME: should not expand. legacy macro scoping is not implemented.
}

#[test]
fn plain_macros_are_legacy_textual_scoped() {
    check(
        r#"
//- /main.rs
mod m1;
bar!(NotFoundNotMacroUse);

mod m2 { foo!(NotFoundBeforeInside2); }

macro_rules! foo {
    ($x:ident) => { struct $x; }
}
foo!(Ok);

mod m3;
foo!(OkShadowStop);
bar!(NotFoundMacroUseStop);

#[macro_use]
mod m5 {
    #[macro_use]
    mod m6 {
        macro_rules! foo {
            ($x:ident) => { fn $x() {} }
        }
    }
}
foo!(ok_double_macro_use_shadow);

baz!(NotFoundBefore);
#[macro_use]
mod m7 {
    macro_rules! baz {
        ($x:ident) => { struct $x; }
    }
}
baz!(OkAfter);

//- /m1.rs
foo!(NotFoundBeforeInside1);
macro_rules! bar {
    ($x:ident) => { struct $x; }
}

//- /m3/mod.rs
foo!(OkAfterInside);
macro_rules! foo {
    ($x:ident) => { fn $x() {} }
}
foo!(ok_shadow);

#[macro_use]
mod m4;
bar!(OkMacroUse);

mod m5;
baz!(OkMacroUseInner);

//- /m3/m4.rs
foo!(ok_shadow_deep);
macro_rules! bar {
    ($x:ident) => { struct $x; }
}
//- /m3/m5.rs
#![macro_use]
macro_rules! baz {
    ($x:ident) => { struct $x; }
}


"#,
        expect![[r#"
            crate
            NotFoundBefore: t v
            Ok: t v
            OkAfter: t v
            OkShadowStop: t v
            m1: t
            m2: t
            m3: t
            m5: t
            m7: t
            ok_double_macro_use_shadow: v

            crate::m1

            crate::m2

            crate::m3
            OkAfterInside: t v
            OkMacroUse: t v
            OkMacroUseInner: t v
            m4: t
            m5: t
            ok_shadow: v

            crate::m3::m4
            ok_shadow_deep: v

            crate::m3::m5

            crate::m5
            m6: t

            crate::m5::m6

            crate::m7
        "#]],
    );
    // FIXME: should not see `NotFoundBefore`
}

#[test]
fn type_value_macro_live_in_different_scopes() {
    check(
        r#"
#[macro_export]
macro_rules! foo {
    ($x:ident) => { type $x = (); }
}

foo!(foo);
use foo as bar;

use self::foo as baz;
fn baz() {}
"#,
        expect![[r#"
            crate
            bar: ti mi
            baz: ti v mi
            foo: t m
        "#]],
    );
}

#[test]
fn macro_use_can_be_aliased() {
    check(
        r#"
//- /main.rs crate:main deps:foo
#[macro_use]
extern crate foo;

foo!(Direct);
bar!(Alias);

//- /lib.rs crate:foo
use crate::foo as bar;

mod m {
    #[macro_export]
    macro_rules! foo {
        ($x:ident) => { struct $x; }
    }
}
"#,
        expect![[r#"
            crate
            Alias: t v
            Direct: t v
            foo: te
        "#]],
    );
}

#[test]
fn path_qualified_macros() {
    check(
        r#"
macro_rules! foo {
    ($x:ident) => { struct $x; }
}

crate::foo!(NotResolved);

crate::bar!(OkCrate);
bar!(OkPlain);
alias1!(NotHere);
m::alias1!(OkAliasPlain);
m::alias2!(OkAliasSuper);
m::alias3!(OkAliasCrate);
not_found!(NotFound);

mod m {
    #[macro_export]
    macro_rules! bar {
        ($x:ident) => { struct $x; }
    }
    pub use bar as alias1;
    pub use super::bar as alias2;
    pub use crate::bar as alias3;
    pub use self::bar as not_found;
}
"#,
        expect![[r#"
            crate
            OkAliasCrate: t v
            OkAliasPlain: t v
            OkAliasSuper: t v
            OkCrate: t v
            OkPlain: t v
            bar: m
            m: t

            crate::m
            alias1: mi
            alias2: mi
            alias3: mi
            not_found: _
        "#]],
    );
}

#[test]
fn macro_dollar_crate_is_correct_in_item() {
    cov_mark::check!(macro_dollar_crate_self);
    check(
        r#"
//- /main.rs crate:main deps:foo
#[macro_use]
extern crate foo;

#[macro_use]
mod m {
    macro_rules! current {
        () => {
            use $crate::Foo as FooSelf;
        }
    }
}

struct Foo;

current!();
not_current1!();
foo::not_current2!();

//- /lib.rs crate:foo
mod m {
    #[macro_export]
    macro_rules! not_current1 {
        () => {
            use $crate::Bar;
        }
    }
}

#[macro_export]
macro_rules! not_current2 {
    () => {
        use $crate::Baz;
    }
}

pub struct Bar;
pub struct Baz;
"#,
        expect![[r#"
            crate
            Bar: ti vi
            Baz: ti vi
            Foo: t v
            FooSelf: ti vi
            foo: te
            m: t

            crate::m
        "#]],
    );
}

#[test]
fn macro_dollar_crate_is_correct_in_indirect_deps() {
    cov_mark::check!(macro_dollar_crate_other);
    // From std
    check(
        r#"
//- /main.rs edition:2018 crate:main deps:std
foo!();

//- /std.rs crate:std deps:core
pub use core::foo;

pub mod prelude {
    pub mod rust_2018 {}
}

#[macro_use]
mod std_macros;

//- /core.rs crate:core
#[macro_export]
macro_rules! foo {
    () => {
        use $crate::bar;
    }
}

pub struct bar;
"#,
        expect![[r#"
            crate
            bar: ti vi
        "#]],
    );
}

#[test]
fn macro_dollar_crate_is_correct_in_derive_meta() {
    compute_crate_def_map(
        r#"
//- minicore: derive, clone
//- /main.rs crate:main deps:lib
lib::foo!();

//- /lib.rs crate:lib
#[macro_export]
macro_rules! foo {
    () => {
        #[derive($crate::Clone)]
        struct S;
    }
}

pub use core::clone::Clone;
"#,
        |map| assert_eq!(map.modules[DefMap::ROOT].scope.impls().len(), 1),
    );
}

#[test]
fn expand_derive() {
    compute_crate_def_map(
        r#"
//- /main.rs crate:main deps:core
use core::Copy;

#[core::derive(Copy, core::Clone)]
struct Foo;

//- /core.rs crate:core
#[rustc_builtin_macro]
pub macro derive($item:item) {}
#[rustc_builtin_macro]
pub macro Copy {}
#[rustc_builtin_macro]
pub macro Clone {}
"#,
        |map| assert_eq!(map.modules[DefMap::ROOT].scope.impls().len(), 2),
    );
}

#[test]
fn resolve_builtin_derive() {
    check(
        r#"
//- /main.rs crate:main deps:core
use core::*;

//- /core.rs crate:core
#[rustc_builtin_macro]
pub macro Clone {}

pub trait Clone {}
"#,
        expect![[r#"
            crate
            Clone: tg mg
        "#]],
    );
}

#[test]
fn builtin_derive_with_unresolved_attributes_fall_back() {
    // Tests that we still resolve derives after ignoring an unresolved attribute.
    cov_mark::check!(unresolved_attribute_fallback);
    compute_crate_def_map(
        r#"
//- /main.rs crate:main deps:core
use core::{Clone, derive};

#[derive(Clone)]
#[unresolved]
struct Foo;

//- /core.rs crate:core
#[rustc_builtin_macro]
pub macro derive($item:item) {}
#[rustc_builtin_macro]
pub macro Clone {}
"#,
        |map| assert_eq!(map.modules[DefMap::ROOT].scope.impls().len(), 1),
    );
}

#[test]
fn unresolved_attributes_fall_back_track_per_file_moditems() {
    // Tests that we track per-file ModItems when ignoring an unresolved attribute.
    // Just tracking the `ModItem` leads to `Foo` getting ignored.

    check(
        r#"
        //- /main.rs crate:main

        mod submod;

        #[unresolved]
        struct Foo;

        //- /submod.rs
        #[unresolved]
        struct Bar;
        "#,
        expect![[r#"
            crate
            Foo: t v
            submod: t

            crate::submod
            Bar: t v
        "#]],
    );
}

#[test]
fn unresolved_attrs_extern_block_hang() {
    // Regression test for https://github.com/rust-lang/rust-analyzer/issues/8905
    check(
        r#"
#[unresolved]
extern "C" {
    #[unresolved]
    fn f();
}
    "#,
        expect![[r#"
        crate
        f: v
    "#]],
    );
}

#[test]
fn macros_in_extern_block() {
    check(
        r#"
macro_rules! m {
    () => { static S: u8; };
}

extern {
    m!();
}
    "#,
        expect![[r#"
            crate
            S: v
        "#]],
    );
}

#[test]
fn resolves_derive_helper() {
    cov_mark::check!(resolved_derive_helper);
    check(
        r#"
//- /main.rs crate:main deps:proc
#[rustc_builtin_macro]
pub macro derive($item:item) {}

#[derive(proc::Derive)]
#[helper]
#[unresolved]
struct S;

//- /proc.rs crate:proc
#![crate_type="proc-macro"]
#[proc_macro_derive(Derive, attributes(helper))]
fn derive() {}
        "#,
        expect![[r#"
            crate
            S: t v
            derive: m
        "#]],
    );
}

#[test]
fn resolves_derive_helper_rustc_builtin_macro() {
    cov_mark::check!(resolved_derive_helper);
    // This is NOT the correct usage of `default` helper attribute, but we don't resolve helper
    // attributes on non mod items in hir nameres.
    check(
        r#"
//- minicore: derive, default
#[derive(Default)]
#[default]
enum E {
    A,
    B,
}
"#,
        expect![[r#"
            crate
            E: t
        "#]],
    );
}

#[test]
fn unresolved_attr_with_cfg_attr_hang() {
    // Another regression test for https://github.com/rust-lang/rust-analyzer/issues/8905
    check(
        r#"
#[cfg_attr(not(off), unresolved, unresolved)]
struct S;
        "#,
        expect![[r#"
            crate
            S: t v
        "#]],
    );
}

#[test]
fn macro_expansion_overflow() {
    cov_mark::check!(macro_expansion_overflow);
    check(
        r#"
macro_rules! a {
    ($e:expr; $($t:tt)*) => {
        b!(static = (); $($t)*);
    };
    () => {};
}

macro_rules! b {
    (static = $e:expr; $($t:tt)*) => {
        a!($e; $($t)*);
    };
    () => {};
}

b! { static = #[] ();}
"#,
        expect![[r#"
            crate
        "#]],
    );
}

#[test]
fn macros_defining_macros() {
    check(
        r#"
macro_rules! item {
    ($item:item) => { $item }
}

item! {
    macro_rules! indirect_macro { () => { struct S {} } }
}

indirect_macro!();
    "#,
        expect![[r#"
            crate
            S: t
        "#]],
    );
}

#[test]
fn resolves_proc_macros() {
    check(
        r#"
#![crate_type="proc-macro"]
struct TokenStream;

#[proc_macro]
pub fn function_like_macro(args: TokenStream) -> TokenStream {
    args
}

#[proc_macro_attribute]
pub fn attribute_macro(_args: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_derive(DummyTrait)]
pub fn derive_macro(_item: TokenStream) -> TokenStream {
    TokenStream
}

#[proc_macro_derive(AnotherTrait, attributes(helper_attr))]
pub fn derive_macro_2(_item: TokenStream) -> TokenStream {
    TokenStream
}
"#,
        expect![[r#"
            crate
            AnotherTrait: m
            DummyTrait: m
            TokenStream: t v
            attribute_macro: v m
            derive_macro: v
            derive_macro_2: v
            function_like_macro: v m
        "#]],
    );
}

#[test]
fn proc_macro_censoring() {
    // Make sure that only proc macros are publicly exported from proc-macro crates.

    check(
        r#"
//- /main.rs crate:main deps:macros
pub use macros::*;

//- /macros.rs crate:macros
#![crate_type="proc-macro"]
pub struct TokenStream;

#[proc_macro]
pub fn function_like_macro(args: TokenStream) -> TokenStream {
    args
}

#[proc_macro_attribute]
pub fn attribute_macro(_args: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_derive(DummyTrait)]
pub fn derive_macro(_item: TokenStream) -> TokenStream {
    TokenStream
}

#[macro_export]
macro_rules! mbe {
    () => {};
}
"#,
        expect![[r#"
            crate
            DummyTrait: mg
            attribute_macro: mg
            function_like_macro: mg
        "#]],
    );
}

#[test]
fn collects_derive_helpers() {
    let db = TestDB::with_files(
        r#"
#![crate_type="proc-macro"]
struct TokenStream;

#[proc_macro_derive(AnotherTrait, attributes(helper_attr))]
pub fn derive_macro_2(_item: TokenStream) -> TokenStream {
    TokenStream
}
"#,
    );
    let krate = *db.all_crates().last().expect("no crate graph present");
    let def_map = crate_def_map(&db, krate);

    assert_eq!(def_map.data.exported_derives.len(), 1);
    match def_map.data.exported_derives.values().next() {
        Some(helpers) => match &**helpers {
            [attr] => assert_eq!(attr.display(&db, Edition::CURRENT).to_string(), "helper_attr"),
            _ => unreachable!(),
        },
        _ => unreachable!(),
    }
}

#[test]
fn resolve_macro_def() {
    check(
        r#"
pub macro structs($($i:ident),*) {
    $(struct $i { field: u32 } )*
}
structs!(Foo);
"#,
        expect![[r#"
            crate
            Foo: t
            structs: m
        "#]],
    );
}

#[test]
fn macro_in_prelude() {
    check(
        r#"
//- /lib.rs edition:2018 crate:lib deps:std
global_asm!();

//- /std.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub macro global_asm() {
            pub struct S;
        }
    }
}
        "#,
        expect![[r#"
            crate
            S: t v
        "#]],
    )
}

#[test]
fn issue9358_bad_macro_stack_overflow() {
    cov_mark::check!(issue9358_bad_macro_stack_overflow);
    check(
        r#"
macro_rules! m {
  ($cond:expr) => { m!($cond, stringify!($cond)) };
  ($cond:expr, $($arg:tt)*) => { $cond };
}
m!(
"#,
        expect![[r#"
            crate
        "#]],
    )
}

#[test]
fn eager_macro_correctly_resolves_contents() {
    // Eager macros resolve any contained macros when expanded. This should work correctly with the
    // usual name resolution rules, so both of these `include!`s should include the right file.

    check(
        r#"
//- /lib.rs
#[rustc_builtin_macro]
macro_rules! include { () => {} }

include!(inner_a!());
include!(crate::inner_b!());

#[macro_export]
macro_rules! inner_a {
    () => { "inc_a.rs" };
}
#[macro_export]
macro_rules! inner_b {
    () => { "inc_b.rs" };
}
//- /inc_a.rs
struct A;
//- /inc_b.rs
struct B;
"#,
        expect![[r#"
        crate
        A: t v
        B: t v
        inner_a: m
        inner_b: m
    "#]],
    );
}

#[test]
fn eager_macro_correctly_resolves_dollar_crate() {
    // MBE -> eager -> $crate::mbe
    check(
        r#"
//- /lib.rs
#[rustc_builtin_macro]
macro_rules! include { () => {} }

#[macro_export]
macro_rules! inner {
    () => { "inc.rs" };
}

macro_rules! m {
    () => { include!($crate::inner!()); };
}

m!();

//- /inc.rs
struct A;
"#,
        expect![[r#"
            crate
            A: t v
            inner: m
        "#]],
    );
    // eager -> MBE -> $crate::mbe
    check(
        r#"
//- /lib.rs
#[rustc_builtin_macro]
macro_rules! include { () => {} }

#[macro_export]
macro_rules! inner {
    () => { "inc.rs" };
}

macro_rules! n {
    () => {
        $crate::inner!()
    };
}

include!(n!());

//- /inc.rs
struct A;
"#,
        expect![[r#"
            crate
            A: t v
            inner: m
        "#]],
    );
}

#[test]
fn nested_include() {
    check(
        r#"
//- minicore: include
//- /lib.rs
include!("out_dir/includes.rs");

//- /out_dir/includes.rs
pub mod company_name {
    pub mod network {
        pub mod v1 {
            include!("company_name.network.v1.rs");
        }
    }
}
//- /out_dir/company_name.network.v1.rs
pub struct IpAddress {
    pub ip_type: &'static str,
}
/// Nested message and enum types in `IpAddress`.
pub mod ip_address {
    pub enum IpType {
        IpV4(u32),
    }
}

"#,
        expect![[r#"
            crate
            company_name: t

            crate::company_name
            network: t

            crate::company_name::network
            v1: t

            crate::company_name::network::v1
            IpAddress: t
            ip_address: t

            crate::company_name::network::v1::ip_address
            IpType: t
        "#]],
    );
}

#[test]
fn include_with_submod_file() {
    check(
        r#"
//- minicore: include
//- /lib.rs
include!("out_dir/includes.rs");

//- /out_dir/includes.rs
pub mod company_name {
    pub mod network {
        pub mod v1;
    }
}
//- /out_dir/company_name/network/v1.rs
pub struct IpAddress {
    pub ip_type: &'static str,
}
/// Nested message and enum types in `IpAddress`.
pub mod ip_address {
    pub enum IpType {
        IpV4(u32),
    }
}

"#,
        expect![[r#"
            crate
            company_name: t

            crate::company_name
            network: t

            crate::company_name::network
            v1: t

            crate::company_name::network::v1
            IpAddress: t
            ip_address: t

            crate::company_name::network::v1::ip_address
            IpType: t
        "#]],
    );
}

#[test]
fn include_many_mods() {
    check(
        r#"
//- /lib.rs
#[rustc_builtin_macro]
macro_rules! include { () => {} }

mod nested {
    include!("out_dir/includes.rs");

    mod different_company {
        include!("out_dir/different_company/mod.rs");
    }

    mod util;
}

//- /nested/util.rs
pub struct Helper {}
//- /out_dir/includes.rs
pub mod company_name {
    pub mod network {
        pub mod v1;
    }
}
//- /out_dir/company_name/network/v1.rs
pub struct IpAddress {}
//- /out_dir/different_company/mod.rs
pub mod network;
//- /out_dir/different_company/network.rs
pub struct Url {}

"#,
        expect![[r#"
            crate
            nested: t

            crate::nested
            company_name: t
            different_company: t
            util: t

            crate::nested::company_name
            network: t

            crate::nested::company_name::network
            v1: t

            crate::nested::company_name::network::v1
            IpAddress: t

            crate::nested::different_company
            network: t

            crate::nested::different_company::network
            Url: t

            crate::nested::util
            Helper: t
        "#]],
    );
}

#[test]
fn macro_use_imports_all_macro_types() {
    let db = TestDB::with_files(
        r#"
//- /main.rs crate:main deps:lib
#[macro_use]
extern crate lib;

//- /lib.rs crate:lib deps:proc
pub use proc::*;

#[macro_export]
macro_rules! legacy { () => () }

pub macro macro20 {}

//- /proc.rs crate:proc
#![crate_type="proc-macro"]

struct TokenStream;

#[proc_macro_attribute]
fn proc_attr(a: TokenStream, b: TokenStream) -> TokenStream { a }
    "#,
    );
    let krate = *db.all_crates().last().expect("no crate graph present");
    let def_map = crate_def_map(&db, krate);

    let root_module = &def_map[DefMap::ROOT].scope;
    assert!(
        root_module.legacy_macros().count() == 0,
        "`#[macro_use]` shouldn't bring macros into textual macro scope",
    );

    let actual = def_map
        .macro_use_prelude
        .keys()
        .map(|name| name.display(&db, Edition::CURRENT).to_string())
        .sorted()
        .join("\n");

    expect![[r#"
        legacy
        macro20
        proc_attr"#]]
    .assert_eq(&actual);
}

#[test]
fn non_prelude_macros_take_precedence_over_macro_use_prelude() {
    check(
        r#"
//- /lib.rs edition:2021 crate:lib deps:dep,core
#[macro_use]
extern crate dep;

macro foo() { struct Ok; }
macro bar() { fn ok() {} }

foo!();
bar!();

//- /dep.rs crate:dep
#[macro_export]
macro_rules! foo {
    () => { struct NotOk; }
}

//- /core.rs crate:core
pub mod prelude {
    pub mod rust_2021 {
        #[macro_export]
        macro_rules! bar {
            () => { fn not_ok() {} }
        }
    }
}
        "#,
        expect![[r#"
            crate
            Ok: t v
            bar: m
            dep: te
            foo: m
            ok: v
        "#]],
    );
}

#[test]
fn macro_use_prelude_is_eagerly_expanded() {
    // See FIXME in `ModCollector::collect_macro_call()`.
    check(
        r#"
//- /main.rs crate:main deps:lib
#[macro_use]
extern crate lib;
mk_foo!();
mod a {
    foo!();
}
//- /lib.rs crate:lib
#[macro_export]
macro_rules! mk_foo {
    () => {
        macro_rules! foo {
            () => { struct Ok; }
        }
    }
}
    "#,
        expect![[r#"
            crate
            a: t
            lib: te

            crate::a
            Ok: t v
        "#]],
    );
}

#[test]
fn macro_sub_namespace() {
    compute_crate_def_map(
        r#"
//- minicore: derive, clone
macro_rules! Clone { () => {} }
macro_rules! derive { () => {} }

#[derive(Clone)]
struct S;
    "#,
        |map| assert_eq!(map.modules[DefMap::ROOT].scope.impls().len(), 1),
    );
}

#[test]
fn macro_sub_namespace2() {
    check(
        r#"
//- /main.rs edition:2021 crate:main deps:proc,core
use proc::{foo, bar};

foo!();
bar!();

//- /proc.rs crate:proc
#![crate_type="proc-macro"]
#[proc_macro_derive(foo)]
pub fn foo() {}
#[proc_macro_attribute]
pub fn bar() {}

//- /core.rs crate:core
pub mod prelude {
    pub mod rust_2021 {
        pub macro foo() {
            struct Ok;
        }
        pub macro bar() {
            fn ok() {}
        }
    }
}
    "#,
        expect![[r#"
            crate
            Ok: t v
            bar: mi
            foo: mi
            ok: v
        "#]],
    );
}
