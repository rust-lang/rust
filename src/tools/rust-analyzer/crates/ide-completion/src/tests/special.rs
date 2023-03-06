//! Tests that don't fit into a specific category.

use expect_test::{expect, Expect};

use crate::tests::{
    check_edit, completion_list, completion_list_no_kw, completion_list_with_trigger_character,
};

fn check_no_kw(ra_fixture: &str, expect: Expect) {
    let actual = completion_list_no_kw(ra_fixture);
    expect.assert_eq(&actual)
}

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual)
}

pub(crate) fn check_with_trigger_character(
    ra_fixture: &str,
    trigger_character: Option<char>,
    expect: Expect,
) {
    let actual = completion_list_with_trigger_character(ra_fixture, trigger_character);
    expect.assert_eq(&actual)
}

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
    _alpha()$0
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
                st Option
                bt u32
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
                fn f()        fn()
                ma concat!(…) macro_rules! concat
                md std
                bt u32
            "#]],
    );
}

#[test]
fn completes_std_prelude_if_core_is_defined() {
    check_no_kw(
        r#"
//- /main.rs crate:main deps:core,std
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
                st String
                bt u32
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
                bt u32
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
                ct PUBLIC_CONST    pub const PUBLIC_CONST: u32
                fn public_method() fn()
                ta PublicType      pub type PublicType = u32
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
                ct C2 (as Sub)           const C2: ()
                ct CONST (as Super)      const CONST: u8
                fn func() (as Super)     fn()
                fn subfunc() (as Sub)    fn()
                ta SubTy (as Sub)        type SubTy
                ta Ty (as Super)         type Ty
                me method(…) (as Super)  fn(&self)
                me submethod(…) (as Sub) fn(&self)
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
                ct C2 (as Sub)           const C2: ()
                ct CONST (as Super)      const CONST: u8
                fn func() (as Super)     fn()
                fn subfunc() (as Sub)    fn()
                ta SubTy (as Sub)        type SubTy
                ta Ty (as Super)         type Ty
                me method(…) (as Super)  fn(&self)
                me submethod(…) (as Sub) fn(&self)
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
                fn main()  fn()
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
                ct RIGHT_CONST
                fn right_fn()  fn()
                st RightType
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
                ev Bar    Bar
                ev Baz    Baz
                me foo(…) fn(self)
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
#[lang = "u8"]
impl u8 {
    pub const MAX: Self = 255;

    pub fn func(self) {}
}
"#,
        expect![[r#"
                ct MAX     pub const MAX: Self
                me func(…) fn(self)
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
            fn bar()          fn()
            fn foo() (as Foo) fn() -> Self
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
            me by_macro() (as MyTrait) fn(&self)
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
            me by_macro() (as MyTrait) fn(&self)
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
            fn main() fn()
            lc foobar i32
            ma x!(…)  macro_rules! x
            bt u32
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
