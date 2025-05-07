//! Tests that don't fit into a specific category.

use expect_test::{Expect, expect};
use ide_db::SymbolKind;

use crate::{
    CompletionItemKind,
    tests::{
        TEST_CONFIG, check, check_edit, check_no_kw, check_with_trigger_character,
        do_completion_with_config,
    },
};

#[test]
fn completes_if_prefix_is_keyword() {
    check_edit(
        "wherewolf",
        r#"
fn main() {
    let wherewolf = 92;
    drop(where$0)
}
"#,
        r#"
fn main() {
    let wherewolf = 92;
    drop(wherewolf)
}
"#,
    )
}

/// Regression test for issue #6091.
#[test]
fn correctly_completes_module_items_prefixed_with_underscore() {
    check_edit(
        "_alpha",
        r#"
fn main() {
    _$0
}
fn _alpha() {}
"#,
        r#"
fn main() {
    _alpha();$0
}
fn _alpha() {}
"#,
    )
}

#[test]
fn completes_prelude() {
    check_no_kw(
        r#"
//- /main.rs edition:2018 crate:main deps:std
fn foo() { let x: $0 }

//- /std/lib.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub struct Option;
    }
}
"#,
        expect![[r#"
            md std
            st Option Option
            bt u32       u32
        "#]],
    );
}

#[test]
fn completes_prelude_macros() {
    check_no_kw(
        r#"
//- /main.rs edition:2018 crate:main deps:std
fn f() {$0}

//- /std/lib.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub use crate::concat;
    }
}

mod macros {
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! concat { }
}
"#,
        expect![[r#"
            fn f()                       fn()
            ma concat!(…) macro_rules! concat
            md std
            bt u32                        u32
        "#]],
    );
}

#[test]
fn completes_std_prelude_if_core_is_defined() {
    check_no_kw(
        r#"
//- /main.rs crate:main deps:core,std edition:2021
fn foo() { let x: $0 }

//- /core/lib.rs crate:core
pub mod prelude {
    pub mod rust_2021 {
        pub struct Option;
    }
}

//- /std/lib.rs crate:std deps:core
pub mod prelude {
    pub mod rust_2021 {
        pub struct String;
    }
}
"#,
        expect![[r#"
            md core
            md std
            st String String
            bt u32       u32
        "#]],
    );
}

#[test]
fn respects_doc_hidden() {
    check_no_kw(
        r#"
//- /lib.rs crate:lib deps:std
fn f() {
    format_$0
}

//- /std.rs crate:std
#[doc(hidden)]
#[macro_export]
macro_rules! format_args_nl {
    () => {}
}

pub mod prelude {
    pub mod rust_2018 {}
}
            "#,
        expect![[r#"
            fn f() fn()
            md std
            bt u32  u32
        "#]],
    );
}

#[test]
fn respects_doc_hidden_in_assoc_item_list() {
    check_no_kw(
        r#"
//- /lib.rs crate:lib deps:std
struct S;
impl S {
    format_$0
}

//- /std.rs crate:std
#[doc(hidden)]
#[macro_export]
macro_rules! format_args_nl {
    () => {}
}

pub mod prelude {
    pub mod rust_2018 {}
}
            "#,
        expect![[r#"
                md std
            "#]],
    );
}

#[test]
fn associated_item_visibility() {
    check_no_kw(
        r#"
//- /lib.rs crate:lib new_source_root:library
pub struct S;

impl S {
    pub fn public_method() { }
    fn private_method() { }
    pub type PublicType = u32;
    type PrivateType = u32;
    pub const PUBLIC_CONST: u32 = 1;
    const PRIVATE_CONST: u32 = 1;
}

//- /main.rs crate:main deps:lib new_source_root:local
fn foo() { let _ = lib::S::$0 }
"#,
        expect![[r#"
            ct PUBLIC_CONST pub const PUBLIC_CONST: u32
            fn public_method()                     fn()
            ta PublicType     pub type PublicType = u32
        "#]],
    );
}

#[test]
fn completes_union_associated_method() {
    check_no_kw(
        r#"
union U {};
impl U { fn m() { } }

fn foo() { let _ = U::$0 }
"#,
        expect![[r#"
            fn m() fn()
        "#]],
    );
}

#[test]
fn completes_trait_associated_method_1() {
    check_no_kw(
        r#"
trait Trait { fn m(); }

fn foo() { let _ = Trait::$0 }
"#,
        expect![[r#"
            fn m() (as Trait) fn()
        "#]],
    );
}

#[test]
fn completes_trait_associated_method_2() {
    check_no_kw(
        r#"
trait Trait { fn m(); }

struct S;
impl Trait for S {}

fn foo() { let _ = S::$0 }
"#,
        expect![[r#"
            fn m() (as Trait) fn()
        "#]],
    );
}

#[test]
fn completes_trait_associated_method_3() {
    check_no_kw(
        r#"
trait Trait { fn m(); }

struct S;
impl Trait for S {}

fn foo() { let _ = <S as Trait>::$0 }
"#,
        expect![[r#"
            fn m() (as Trait) fn()
        "#]],
    );
}

#[test]
fn completes_ty_param_assoc_ty() {
    check_no_kw(
        r#"
trait Super {
    type Ty;
    const CONST: u8;
    fn func() {}
    fn method(&self) {}
}

trait Sub: Super {
    type SubTy;
    const C2: ();
    fn subfunc() {}
    fn submethod(&self) {}
}

fn foo<T: Sub>() { T::$0 }
"#,
        expect![[r#"
            ct C2 (as Sub)         const C2: ()
            ct CONST (as Super) const CONST: u8
            fn func() (as Super)           fn()
            fn subfunc() (as Sub)          fn()
            me method(…) (as Super)   fn(&self)
            me submethod(…) (as Sub)  fn(&self)
            ta SubTy (as Sub)        type SubTy
            ta Ty (as Super)            type Ty
        "#]],
    );
}

#[test]
fn completes_self_param_assoc_ty() {
    check_no_kw(
        r#"
trait Super {
    type Ty;
    const CONST: u8 = 0;
    fn func() {}
    fn method(&self) {}
}

trait Sub: Super {
    type SubTy;
    const C2: () = ();
    fn subfunc() {}
    fn submethod(&self) {}
}

struct Wrap<T>(T);
impl<T> Super for Wrap<T> {}
impl<T> Sub for Wrap<T> {
    fn subfunc() {
        // Should be able to assume `Self: Sub + Super`
        Self::$0
    }
}
"#,
        expect![[r#"
            ct C2 (as Sub)         const C2: ()
            ct CONST (as Super) const CONST: u8
            fn func() (as Super)           fn()
            fn subfunc() (as Sub)          fn()
            me method(…) (as Super)   fn(&self)
            me submethod(…) (as Sub)  fn(&self)
            ta SubTy (as Sub)        type SubTy
            ta Ty (as Super)            type Ty
        "#]],
    );
}

#[test]
fn completes_type_alias() {
    check_no_kw(
        r#"
struct S;
impl S { fn foo() {} }
type T = S;
impl T { fn bar() {} }

fn main() { T::$0; }
"#,
        expect![[r#"
            fn bar() fn()
            fn foo() fn()
        "#]],
    );
}

#[test]
fn completes_qualified_macros() {
    check_no_kw(
        r#"
#[macro_export]
macro_rules! foo { () => {} }

fn main() { let _ = crate::$0 }
"#,
        expect![[r#"
            fn main()              fn()
            ma foo!(…) macro_rules! foo
        "#]],
    );
}

#[test]
fn does_not_complete_non_fn_macros() {
    check_no_kw(
        r#"
mod m {
    #[rustc_builtin_macro]
    pub macro Clone {}
}

fn f() {m::$0}
"#,
        expect![[r#""#]],
    );
    check_no_kw(
        r#"
mod m {
    #[rustc_builtin_macro]
    pub macro bench {}
}

fn f() {m::$0}
"#,
        expect![[r#""#]],
    );
}

#[test]
fn completes_reexported_items_under_correct_name() {
    check_no_kw(
        r#"
fn foo() { self::m::$0 }

mod m {
    pub use super::p::wrong_fn as right_fn;
    pub use super::p::WRONG_CONST as RIGHT_CONST;
    pub use super::p::WrongType as RightType;
}
mod p {
    pub fn wrong_fn() {}
    pub const WRONG_CONST: u32 = 1;
    pub struct WrongType {};
}
"#,
        expect![[r#"
            ct RIGHT_CONST     u32
            fn right_fn()     fn()
            st RightType WrongType
        "#]],
    );

    check_edit(
        "RightType",
        r#"
fn foo() { self::m::$0 }

mod m {
    pub use super::p::wrong_fn as right_fn;
    pub use super::p::WRONG_CONST as RIGHT_CONST;
    pub use super::p::WrongType as RightType;
}
mod p {
    pub fn wrong_fn() {}
    pub const WRONG_CONST: u32 = 1;
    pub struct WrongType {};
}
"#,
        r#"
fn foo() { self::m::RightType }

mod m {
    pub use super::p::wrong_fn as right_fn;
    pub use super::p::WRONG_CONST as RIGHT_CONST;
    pub use super::p::WrongType as RightType;
}
mod p {
    pub fn wrong_fn() {}
    pub const WRONG_CONST: u32 = 1;
    pub struct WrongType {};
}
"#,
    );
}

#[test]
fn completes_in_simple_macro_call() {
    check_no_kw(
        r#"
macro_rules! m { ($e:expr) => { $e } }
fn main() { m!(self::f$0); }
fn foo() {}
"#,
        expect![[r#"
            fn foo()  fn()
            fn main() fn()
        "#]],
    );
}

#[test]
fn function_mod_share_name() {
    check_no_kw(
        r#"
fn foo() { self::m::$0 }

mod m {
    pub mod z {}
    pub fn z() {}
}
"#,
        expect![[r#"
            fn z() fn()
            md z
        "#]],
    );
}

#[test]
fn completes_hashmap_new() {
    check_no_kw(
        r#"
struct RandomState;
struct HashMap<K, V, S = RandomState> {}

impl<K, V> HashMap<K, V, RandomState> {
    pub fn new() -> HashMap<K, V, RandomState> { }
}
fn foo() {
    HashMap::$0
}
"#,
        expect![[r#"
            fn new() fn() -> HashMap<K, V, RandomState>
        "#]],
    );
}

#[test]
fn completes_variant_through_self() {
    cov_mark::check!(completes_variant_through_self);
    check_no_kw(
        r#"
enum Foo {
    Bar,
    Baz,
}

impl Foo {
    fn foo(self) {
        Self::$0
    }
}
"#,
        expect![[r#"
            me foo(…) fn(self)
            ev Bar         Bar
            ev Baz         Baz
        "#]],
    );
}

#[test]
fn completes_non_exhaustive_variant_within_the_defining_crate() {
    check_no_kw(
        r#"
enum Foo {
    #[non_exhaustive]
    Bar,
    Baz,
}

fn foo(self) {
    Foo::$0
}
"#,
        expect![[r#"
            ev Bar Bar
            ev Baz Baz
        "#]],
    );

    check_no_kw(
        r#"
//- /main.rs crate:main deps:e
fn foo(self) {
    e::Foo::$0
}

//- /e.rs crate:e
enum Foo {
    #[non_exhaustive]
    Bar,
    Baz,
}
"#,
        expect![[r#"
            ev Baz Baz
        "#]],
    );
}

#[test]
fn completes_primitive_assoc_const() {
    cov_mark::check!(completes_primitive_assoc_const);
    check_no_kw(
        r#"
//- /lib.rs crate:lib deps:core
fn f() {
    u8::$0
}

//- /core.rs crate:core
#![rustc_coherence_is_core]
#[lang = "u8"]
impl u8 {
    pub const MAX: Self = 255;

    pub fn func(self) {}
}
"#,
        expect![[r#"
            ct MAX pub const MAX: Self
            me func(…)        fn(self)
        "#]],
    );
}

#[test]
fn completes_variant_through_alias() {
    cov_mark::check!(completes_variant_through_alias);
    check_no_kw(
        r#"
enum Foo {
    Bar
}
type Foo2 = Foo;
fn main() {
    Foo2::$0
}
"#,
        expect![[r#"
            ev Bar Bar
        "#]],
    );
}

#[test]
fn respects_doc_hidden2() {
    check_no_kw(
        r#"
//- /lib.rs crate:lib deps:dep
fn f() {
    dep::$0
}

//- /dep.rs crate:dep
#[doc(hidden)]
#[macro_export]
macro_rules! m {
    () => {}
}

#[doc(hidden)]
pub fn f() {}

#[doc(hidden)]
pub struct S;

#[doc(hidden)]
pub mod m {}
            "#,
        expect![[r#""#]],
    )
}

#[test]
fn type_anchor_empty() {
    check_no_kw(
        r#"
trait Foo {
    fn foo() -> Self;
}
struct Bar;
impl Foo for Bar {
    fn foo() -> {
        Bar
    }
}
fn bar() -> Bar {
    <_>::$0
}
"#,
        expect![[r#"
            fn foo() (as Foo) fn() -> Self
            ex Bar
            ex bar()
        "#]],
    );
}

#[test]
fn type_anchor_type() {
    check_no_kw(
        r#"
trait Foo {
    fn foo() -> Self;
}
struct Bar;
impl Bar {
    fn bar() {}
}
impl Foo for Bar {
    fn foo() -> {
        Bar
    }
}
fn bar() -> Bar {
    <Bar>::$0
}
"#,
        expect![[r#"
            fn bar()                  fn()
            fn foo() (as Foo) fn() -> Self
            ex Bar
            ex bar()
        "#]],
    );
}

#[test]
fn type_anchor_type_trait() {
    check_no_kw(
        r#"
trait Foo {
    fn foo() -> Self;
}
struct Bar;
impl Bar {
    fn bar() {}
}
impl Foo for Bar {
    fn foo() -> {
        Bar
    }
}
fn bar() -> Bar {
    <Bar as Foo>::$0
}
"#,
        expect![[r#"
            fn foo() (as Foo) fn() -> Self
            ex Bar
            ex bar()
        "#]],
    );
}

#[test]
fn completes_fn_in_pub_trait_generated_by_macro() {
    check_no_kw(
        r#"
mod other_mod {
    macro_rules! make_method {
        ($name:ident) => {
            fn $name(&self) {}
        };
    }

    pub trait MyTrait {
        make_method! { by_macro }
        fn not_by_macro(&self) {}
    }

    pub struct Foo {}

    impl MyTrait for Foo {}
}

fn main() {
    use other_mod::{Foo, MyTrait};
    let f = Foo {};
    f.$0
}
"#,
        expect![[r#"
            me by_macro() (as MyTrait)     fn(&self)
            me not_by_macro() (as MyTrait) fn(&self)
        "#]],
    )
}

#[test]
fn completes_fn_in_pub_trait_generated_by_recursive_macro() {
    check_no_kw(
        r#"
mod other_mod {
    macro_rules! make_method {
        ($name:ident) => {
            fn $name(&self) {}
        };
    }

    macro_rules! make_trait {
        () => {
            pub trait MyTrait {
                make_method! { by_macro }
                fn not_by_macro(&self) {}
            }
        }
    }

    make_trait!();

    pub struct Foo {}

    impl MyTrait for Foo {}
}

fn main() {
    use other_mod::{Foo, MyTrait};
    let f = Foo {};
    f.$0
}
"#,
        expect![[r#"
            me by_macro() (as MyTrait)     fn(&self)
            me not_by_macro() (as MyTrait) fn(&self)
        "#]],
    )
}

#[test]
fn completes_const_in_pub_trait_generated_by_macro() {
    check_no_kw(
        r#"
mod other_mod {
    macro_rules! make_const {
        ($name:ident) => {
            const $name: u8 = 1;
        };
    }

    pub trait MyTrait {
        make_const! { by_macro }
    }

    pub struct Foo {}

    impl MyTrait for Foo {}
}

fn main() {
    use other_mod::{Foo, MyTrait};
    let f = Foo {};
    Foo::$0
}
"#,
        expect![[r#"
            ct by_macro (as MyTrait) pub const by_macro: u8
        "#]],
    )
}

#[test]
fn completes_locals_from_macros() {
    check_no_kw(
        r#"

macro_rules! x {
    ($x:ident, $expr:expr) => {
        let $x = 0;
        $expr
    };
}
fn main() {
    x! {
        foobar, {
            f$0
        }
    };
}
"#,
        expect![[r#"
            fn main()          fn()
            lc foobar           i32
            ma x!(…) macro_rules! x
            bt u32              u32
        "#]],
    )
}

#[test]
fn regression_12644() {
    check_no_kw(
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
    vec![Struct].$0;
}
"#,
        expect![[r#"
            me foo() fn(self)
        "#]],
    );
}

#[test]
fn completes_after_colon_with_trigger() {
    check_with_trigger_character(
        r#"
//- minicore: option
fn foo { ::$0 }
"#,
        Some(':'),
        expect![[r#"
            md core
        "#]],
    );
    check_with_trigger_character(
        r#"
//- minicore: option
fn foo { /* test */::$0 }
"#,
        Some(':'),
        expect![[r#"
            md core
        "#]],
    );

    check_with_trigger_character(
        r#"
fn foo { crate::$0 }
"#,
        Some(':'),
        expect![[r#"
            fn foo() fn()
        "#]],
    );

    check_with_trigger_character(
        r#"
fn foo { crate:$0 }
"#,
        Some(':'),
        expect![""],
    );
}

#[test]
fn completes_after_colon_without_trigger() {
    check_with_trigger_character(
        r#"
fn foo { crate::$0 }
"#,
        None,
        expect![[r#"
            fn foo() fn()
        "#]],
    );

    check_with_trigger_character(
        r#"
fn foo { crate:$0 }
"#,
        None,
        expect![""],
    );
}

#[test]
fn no_completions_in_invalid_path() {
    check(
        r#"
fn foo { crate:::$0 }
"#,
        expect![""],
    );
    check_no_kw(
        r#"
fn foo { crate::::$0 }
"#,
        expect![""],
    )
}

#[test]
fn completes_struct_via_doc_alias_in_fn_body() {
    check(
        r#"
#[doc(alias = "Bar")]
struct Foo;

fn here_we_go() {
    $0
}
"#,
        expect![[r#"
            fn here_we_go()   fn()
            st Foo (alias Bar) Foo
            bt u32             u32
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
fn completes_struct_via_multiple_doc_aliases_in_fn_body() {
    check(
        r#"
#[doc(alias("Bar", "Qux"))]
#[doc(alias = "Baz")]
struct Foo;

fn here_we_go() {
    B$0
}
"#,
        expect![[r#"
            fn here_we_go()             fn()
            st Foo (alias Bar, Qux, Baz) Foo
            bt u32                       u32
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
fn completes_field_name_via_doc_alias_in_fn_body() {
    check(
        r#"
struct Foo {
    #[doc(alias = "qux")]
    bar: u8
};

fn here_we_go() {
    let foo = Foo { q$0 }
}
"#,
        expect![[r#"
            fd bar (alias qux) u8
        "#]],
    );
}

#[test]
fn completes_struct_fn_name_via_doc_alias_in_fn_body() {
    check(
        r#"
struct Foo;
impl Foo {
    #[doc(alias = "qux")]
    fn bar() -> u8 { 1 }
}

fn here_we_go() {
    Foo::q$0
}
"#,
        expect![[r#"
            fn bar() (alias qux) fn() -> u8
        "#]],
    );
}

#[test]
fn completes_method_name_via_doc_alias_in_fn_body() {
    check(
        r#"
struct Foo {
    bar: u8
}
impl Foo {
    #[doc(alias = "qux")]
    fn baz(&self) -> u8 {
        self.bar
    }
}

fn here_we_go() {
    let foo = Foo { field: 42 };
    foo.q$0
}
"#,
        expect![[r#"
            fd bar                            u8
            me baz() (alias qux) fn(&self) -> u8
            sn box                Box::new(expr)
            sn call               function(expr)
            sn const                    const {}
            sn dbg                    dbg!(expr)
            sn dbgr                  dbg!(&expr)
            sn deref                       *expr
            sn let                           let
            sn letm                      let mut
            sn match               match expr {}
            sn ref                         &expr
            sn refm                    &mut expr
            sn return                return expr
            sn unsafe                  unsafe {}
        "#]],
    );
}

#[test]
fn completes_fn_name_via_doc_alias_in_fn_body() {
    check(
        r#"
#[doc(alias = "qux")]
fn foo() {}
fn bar() { qu$0 }
"#,
        expect![[r#"
            fn bar()             fn()
            fn foo() (alias qux) fn()
            bt u32                u32
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
fn completes_struct_name_via_doc_alias_in_another_mod() {
    check(
        r#"
mod foo {
    #[doc(alias = "Qux")]
    pub struct Bar(u8);
}

fn here_we_go() {
    use foo;
    let foo = foo::Q$0
}
"#,
        expect![[r#"
            st Bar (alias Qux) Bar
        "#]],
    );
}

#[test]
fn completes_use_via_doc_alias_in_another_mod() {
    check(
        r#"
mod foo {
    #[doc(alias = "Qux")]
    pub struct Bar(u8);
}

fn here_we_go() {
    use foo::Q$0;
}
"#,
        expect![[r#"
            st Bar (alias Qux) Bar
        "#]],
    );
}

#[test]
fn completes_flyimport_with_doc_alias_in_another_mod() {
    check(
        r#"
mod foo {
    #[doc(alias = "Qux")]
    pub struct Bar();
}

fn here_we_go() {
    let foo = Bar$0
}
"#,
        expect![[r#"
            fn here_we_go()                  fn()
            md foo
            st Bar (alias Qux) (use foo::Bar) Bar
            bt u32                            u32
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
fn completes_only_public() {
    check(
        r#"
//- /e.rs
pub(self) fn i_should_be_hidden() {}
pub(in crate::e) fn i_should_also_be_hidden() {}
pub fn i_am_public () {}

//- /lib.rs crate:krate
pub mod e;

//- /main.rs deps:krate crate:main
use krate::e;
fn main() {
    e::$0
}"#,
        expect![[r#"
            fn i_am_public() fn()
        "#]],
    )
}

#[test]
fn completion_filtering_excludes_non_identifier_doc_aliases() {
    check_edit(
        "PartialOrdcmporder",
        r#"
#[doc(alias = ">")]
#[doc(alias = "cmp")]
#[doc(alias = "order")]
trait PartialOrd {}

struct Foo<T: Partial$0
"#,
        r#"
#[doc(alias = ">")]
#[doc(alias = "cmp")]
#[doc(alias = "order")]
trait PartialOrd {}

struct Foo<T: PartialOrd
"#,
    );
}

fn check_signatures(src: &str, kind: CompletionItemKind, reduced: Expect, full: Expect) {
    const FULL_SIGNATURES_CONFIG: crate::CompletionConfig<'_> = {
        let mut x = TEST_CONFIG;
        x.full_function_signatures = true;
        x
    };

    // reduced signature
    let completion = do_completion_with_config(TEST_CONFIG, src, kind);
    assert!(completion[0].detail.is_some());
    reduced.assert_eq(completion[0].detail.as_ref().unwrap());

    // full signature
    let completion = do_completion_with_config(FULL_SIGNATURES_CONFIG, src, kind);
    assert!(completion[0].detail.is_some());
    full.assert_eq(completion[0].detail.as_ref().unwrap());
}

#[test]
fn respects_full_function_signatures() {
    check_signatures(
        r#"
pub fn foo<'x, T>(x: &'x mut T) -> u8 where T: Clone, { 0u8 }
fn main() { fo$0 }
"#,
        CompletionItemKind::SymbolKind(ide_db::SymbolKind::Function),
        expect!("fn(&'x mut T) -> u8"),
        expect!("pub fn foo<'x, T>(x: &'x mut T) -> u8 where T: Clone,"),
    );

    check_signatures(
        r#"
struct Foo;
struct Bar;
impl Bar {
    pub const fn baz(x: Foo) -> ! { loop {} };
}

fn main() { Bar::b$0 }
"#,
        CompletionItemKind::SymbolKind(ide_db::SymbolKind::Function),
        expect!("const fn(Foo) -> !"),
        expect!("pub const fn baz(x: Foo) -> !"),
    );

    check_signatures(
        r#"
struct Foo;
struct Bar;
impl Bar {
    pub const fn baz<'foo>(&'foo mut self, x: &'foo Foo) -> ! { loop {} };
}

fn main() {
    let mut bar = Bar;
    bar.b$0
}
"#,
        CompletionItemKind::SymbolKind(SymbolKind::Method),
        expect!("const fn(&'foo mut self, &'foo Foo) -> !"),
        expect!("pub const fn baz<'foo>(&'foo mut self, x: &'foo Foo) -> !"),
    );
}

#[test]
fn skips_underscore() {
    check_with_trigger_character(
        r#"
fn foo(_$0) { }
"#,
        Some('_'),
        expect![[r#""#]],
    );
    check_with_trigger_character(
        r#"
fn foo(_: _$0) { }
"#,
        Some('_'),
        expect![[r#""#]],
    );
    check_with_trigger_character(
        r#"
fn foo<T>() {
    foo::<_$0>();
}
"#,
        Some('_'),
        expect![[r#""#]],
    );
    // underscore expressions are fine, they are invalid so the user definitely meant to type an
    // underscored name here
    check_with_trigger_character(
        r#"
fn foo() {
    _$0
}
"#,
        Some('_'),
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
        "#]],
    );
}

#[test]
fn no_skip_underscore_ident() {
    check_with_trigger_character(
        r#"
fn foo(a_$0) { }
"#,
        Some('_'),
        expect![[r#"
            kw mut
            kw ref
        "#]],
    );
    check_with_trigger_character(
        r#"
fn foo(_: a_$0) { }
"#,
        Some('_'),
        expect![[r#"
            bt u32 u32
            kw crate::
            kw self::
        "#]],
    );
    check_with_trigger_character(
        r#"
fn foo<T>() {
    foo::<a_$0>();
}
"#,
        Some('_'),
        expect![[r#"
            tp T
            bt u32 u32
            kw crate::
            kw self::
        "#]],
    );
}
