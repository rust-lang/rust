//! Completion of paths, i.e. `some::prefix::<|>`.

use hir::{Adt, HasVisibility, PathResolution, ScopeDef};
use rustc_hash::FxHashSet;
use syntax::AstNode;
use test_utils::mark;

use crate::completion::{CompletionContext, Completions};

pub(super) fn complete_qualified_path(acc: &mut Completions, ctx: &CompletionContext) {
    let path = match &ctx.path_qual {
        Some(path) => path.clone(),
        None => return,
    };

    if ctx.attribute_under_caret.is_some() || ctx.mod_declaration_under_caret.is_some() {
        return;
    }

    let context_module = ctx.scope.module();

    let resolution = match ctx.sema.resolve_path(&path) {
        Some(res) => res,
        None => return,
    };

    // Add associated types on type parameters and `Self`.
    resolution.assoc_type_shorthand_candidates(ctx.db, |alias| {
        acc.add_type_alias(ctx, alias);
        None::<()>
    });

    match resolution {
        PathResolution::Def(hir::ModuleDef::Module(module)) => {
            let module_scope = module.scope(ctx.db, context_module);
            for (name, def) in module_scope {
                if ctx.use_item_syntax.is_some() {
                    if let ScopeDef::Unknown = def {
                        if let Some(name_ref) = ctx.name_ref_syntax.as_ref() {
                            if name_ref.syntax().text() == name.to_string().as_str() {
                                // for `use self::foo<|>`, don't suggest `foo` as a completion
                                mark::hit!(dont_complete_current_use);
                                continue;
                            }
                        }
                    }
                }

                acc.add_resolution(ctx, name.to_string(), &def);
            }
        }
        PathResolution::Def(def @ hir::ModuleDef::Adt(_))
        | PathResolution::Def(def @ hir::ModuleDef::TypeAlias(_)) => {
            if let hir::ModuleDef::Adt(Adt::Enum(e)) = def {
                for variant in e.variants(ctx.db) {
                    acc.add_enum_variant(ctx, variant, None);
                }
            }
            let ty = match def {
                hir::ModuleDef::Adt(adt) => adt.ty(ctx.db),
                hir::ModuleDef::TypeAlias(a) => a.ty(ctx.db),
                _ => unreachable!(),
            };

            // XXX: For parity with Rust bug #22519, this does not complete Ty::AssocType.
            // (where AssocType is defined on a trait, not an inherent impl)

            let krate = ctx.krate;
            if let Some(krate) = krate {
                let traits_in_scope = ctx.scope.traits_in_scope();
                ty.iterate_path_candidates(ctx.db, krate, &traits_in_scope, None, |_ty, item| {
                    if context_module.map_or(false, |m| !item.is_visible_from(ctx.db, m)) {
                        return None;
                    }
                    match item {
                        hir::AssocItem::Function(func) => {
                            acc.add_function(ctx, func, None);
                        }
                        hir::AssocItem::Const(ct) => acc.add_const(ctx, ct),
                        hir::AssocItem::TypeAlias(ty) => acc.add_type_alias(ctx, ty),
                    }
                    None::<()>
                });

                // Iterate assoc types separately
                ty.iterate_assoc_items(ctx.db, krate, |item| {
                    if context_module.map_or(false, |m| !item.is_visible_from(ctx.db, m)) {
                        return None;
                    }
                    match item {
                        hir::AssocItem::Function(_) | hir::AssocItem::Const(_) => {}
                        hir::AssocItem::TypeAlias(ty) => acc.add_type_alias(ctx, ty),
                    }
                    None::<()>
                });
            }
        }
        PathResolution::Def(hir::ModuleDef::Trait(t)) => {
            // Handles `Trait::assoc` as well as `<Ty as Trait>::assoc`.
            for item in t.items(ctx.db) {
                if context_module.map_or(false, |m| !item.is_visible_from(ctx.db, m)) {
                    continue;
                }
                match item {
                    hir::AssocItem::Function(func) => {
                        acc.add_function(ctx, func, None);
                    }
                    hir::AssocItem::Const(ct) => acc.add_const(ctx, ct),
                    hir::AssocItem::TypeAlias(ty) => acc.add_type_alias(ctx, ty),
                }
            }
        }
        PathResolution::TypeParam(_) | PathResolution::SelfType(_) => {
            if let Some(krate) = ctx.krate {
                let ty = match resolution {
                    PathResolution::TypeParam(param) => param.ty(ctx.db),
                    PathResolution::SelfType(impl_def) => impl_def.target_ty(ctx.db),
                    _ => return,
                };

                let traits_in_scope = ctx.scope.traits_in_scope();
                let mut seen = FxHashSet::default();
                ty.iterate_path_candidates(ctx.db, krate, &traits_in_scope, None, |_ty, item| {
                    if context_module.map_or(false, |m| !item.is_visible_from(ctx.db, m)) {
                        return None;
                    }

                    // We might iterate candidates of a trait multiple times here, so deduplicate
                    // them.
                    if seen.insert(item) {
                        match item {
                            hir::AssocItem::Function(func) => {
                                acc.add_function(ctx, func, None);
                            }
                            hir::AssocItem::Const(ct) => acc.add_const(ctx, ct),
                            hir::AssocItem::TypeAlias(ty) => acc.add_type_alias(ctx, ty),
                        }
                    }
                    None::<()>
                });
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use test_utils::mark;

    use crate::completion::{
        test_utils::{check_edit, completion_list},
        CompletionKind,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Reference);
        expect.assert_eq(&actual);
    }

    fn check_builtin(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::BuiltinType);
        expect.assert_eq(&actual);
    }

    #[test]
    fn dont_complete_current_use() {
        mark::check!(dont_complete_current_use);
        check(r#"use self::foo<|>;"#, expect![[""]]);
    }

    #[test]
    fn dont_complete_current_use_in_braces_with_glob() {
        check(
            r#"
mod foo { pub struct S; }
use self::{foo::*, bar<|>};
"#,
            expect![[r#"
                st S
                md foo
            "#]],
        );
    }

    #[test]
    fn dont_complete_primitive_in_use() {
        check_builtin(r#"use self::<|>;"#, expect![[""]]);
    }

    #[test]
    fn dont_complete_primitive_in_module_scope() {
        check_builtin(r#"fn foo() { self::<|> }"#, expect![[""]]);
    }

    #[test]
    fn completes_primitives() {
        check_builtin(
            r#"fn main() { let _: <|> = 92; }"#,
            expect![[r#"
                bt bool
                bt char
                bt f32
                bt f64
                bt i128
                bt i16
                bt i32
                bt i64
                bt i8
                bt isize
                bt str
                bt u128
                bt u16
                bt u32
                bt u64
                bt u8
                bt usize
            "#]],
        );
    }

    #[test]
    fn completes_mod_with_same_name_as_function() {
        check(
            r#"
use self::my::<|>;

mod my { pub struct Bar; }
fn my() {}
"#,
            expect![[r#"
                st Bar
            "#]],
        );
    }

    #[test]
    fn filters_visibility() {
        check(
            r#"
use self::my::<|>;

mod my {
    struct Bar;
    pub struct Foo;
    pub use Bar as PublicBar;
}
"#,
            expect![[r#"
                st Foo
                st PublicBar
            "#]],
        );
    }

    #[test]
    fn completes_use_item_starting_with_self() {
        check(
            r#"
use self::m::<|>;

mod m { pub struct Bar; }
"#,
            expect![[r#"
                st Bar
            "#]],
        );
    }

    #[test]
    fn completes_use_item_starting_with_crate() {
        check(
            r#"
//- /lib.rs
mod foo;
struct Spam;
//- /foo.rs
use crate::Sp<|>
"#,
            expect![[r#"
                st Spam
                md foo
            "#]],
        );
    }

    #[test]
    fn completes_nested_use_tree() {
        check(
            r#"
//- /lib.rs
mod foo;
struct Spam;
//- /foo.rs
use crate::{Sp<|>};
"#,
            expect![[r#"
                st Spam
                md foo
            "#]],
        );
    }

    #[test]
    fn completes_deeply_nested_use_tree() {
        check(
            r#"
//- /lib.rs
mod foo;
pub mod bar {
    pub mod baz {
        pub struct Spam;
    }
}
//- /foo.rs
use crate::{bar::{baz::Sp<|>}};
"#,
            expect![[r#"
                st Spam
            "#]],
        );
    }

    #[test]
    fn completes_enum_variant() {
        check(
            r#"
enum E { Foo, Bar(i32) }
fn foo() { let _ = E::<|> }
"#,
            expect![[r#"
                ev Bar(…) (i32)
                ev Foo    ()
            "#]],
        );
    }

    #[test]
    fn completes_struct_associated_items() {
        check(
            r#"
//- /lib.rs
struct S;

impl S {
    fn a() {}
    fn b(&self) {}
    const C: i32 = 42;
    type T = i32;
}

fn foo() { let _ = S::<|> }
"#,
            expect![[r#"
                ct C   const C: i32 = 42;
                ta T   type T = i32;
                fn a() fn a()
                me b() fn b(&self)
            "#]],
        );
    }

    #[test]
    fn associated_item_visibility() {
        check(
            r#"
struct S;

mod m {
    impl super::S {
        pub(super) fn public_method() { }
        fn private_method() { }
        pub(super) type PublicType = u32;
        type PrivateType = u32;
        pub(super) const PUBLIC_CONST: u32 = 1;
        const PRIVATE_CONST: u32 = 1;
    }
}

fn foo() { let _ = S::<|> }
"#,
            expect![[r#"
                ct PUBLIC_CONST    pub(super) const PUBLIC_CONST: u32 = 1;
                ta PublicType      pub(super) type PublicType = u32;
                fn public_method() pub(super) fn public_method()
            "#]],
        );
    }

    #[test]
    fn completes_enum_associated_method() {
        check(
            r#"
enum E {};
impl E { fn m() { } }

fn foo() { let _ = E::<|> }
        "#,
            expect![[r#"
                fn m() fn m()
            "#]],
        );
    }

    #[test]
    fn completes_union_associated_method() {
        check(
            r#"
union U {};
impl U { fn m() { } }

fn foo() { let _ = U::<|> }
"#,
            expect![[r#"
                fn m() fn m()
            "#]],
        );
    }

    #[test]
    fn completes_use_paths_across_crates() {
        check(
            r#"
//- /main.rs
use foo::<|>;

//- /foo/lib.rs
pub mod bar { pub struct S; }
"#,
            expect![[r#"
                md bar
            "#]],
        );
    }

    #[test]
    fn completes_trait_associated_method_1() {
        check(
            r#"
trait Trait { fn m(); }

fn foo() { let _ = Trait::<|> }
"#,
            expect![[r#"
                fn m() fn m()
            "#]],
        );
    }

    #[test]
    fn completes_trait_associated_method_2() {
        check(
            r#"
trait Trait { fn m(); }

struct S;
impl Trait for S {}

fn foo() { let _ = S::<|> }
"#,
            expect![[r#"
                fn m() fn m()
            "#]],
        );
    }

    #[test]
    fn completes_trait_associated_method_3() {
        check(
            r#"
trait Trait { fn m(); }

struct S;
impl Trait for S {}

fn foo() { let _ = <S as Trait>::<|> }
"#,
            expect![[r#"
                fn m() fn m()
            "#]],
        );
    }

    #[test]
    fn completes_ty_param_assoc_ty() {
        check(
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

fn foo<T: Sub>() { T::<|> }
"#,
            expect![[r#"
                ct C2          const C2: ();
                ct CONST       const CONST: u8;
                ta SubTy       type SubTy;
                ta Ty          type Ty;
                fn func()      fn func()
                me method()    fn method(&self)
                fn subfunc()   fn subfunc()
                me submethod() fn submethod(&self)
            "#]],
        );
    }

    #[test]
    fn completes_self_param_assoc_ty() {
        check(
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
        Self::<|>
    }
}
"#,
            expect![[r#"
                ct C2          const C2: () = ();
                ct CONST       const CONST: u8 = 0;
                ta SubTy       type SubTy;
                ta Ty          type Ty;
                fn func()      fn func()
                me method()    fn method(&self)
                fn subfunc()   fn subfunc()
                me submethod() fn submethod(&self)
            "#]],
        );
    }

    #[test]
    fn completes_type_alias() {
        check(
            r#"
struct S;
impl S { fn foo() {} }
type T = S;
impl T { fn bar() {} }

fn main() { T::<|>; }
"#,
            expect![[r#"
                fn bar() fn bar()
                fn foo() fn foo()
            "#]],
        );
    }

    #[test]
    fn completes_qualified_macros() {
        check(
            r#"
#[macro_export]
macro_rules! foo { () => {} }

fn main() { let _ = crate::<|> }
        "#,
            expect![[r##"
                ma foo!(…) #[macro_export]
                macro_rules! foo
                fn main()  fn main()
            "##]],
        );
    }

    #[test]
    fn test_super_super_completion() {
        check(
            r#"
mod a {
    const A: usize = 0;
    mod b {
        const B: usize = 0;
        mod c { use super::super::<|> }
    }
}
"#,
            expect![[r#"
                ct A
                md b
            "#]],
        );
    }

    #[test]
    fn completes_reexported_items_under_correct_name() {
        check(
            r#"
fn foo() { self::m::<|> }

mod m {
    pub use super::p::wrong_fn as right_fn;
    pub use super::p::WRONG_CONST as RIGHT_CONST;
    pub use super::p::WrongType as RightType;
}
mod p {
    fn wrong_fn() {}
    const WRONG_CONST: u32 = 1;
    struct WrongType {};
}
"#,
            expect![[r#"
                ct RIGHT_CONST
                st RightType
                fn right_fn()  fn wrong_fn()
            "#]],
        );

        check_edit(
            "RightType",
            r#"
fn foo() { self::m::<|> }

mod m {
    pub use super::p::wrong_fn as right_fn;
    pub use super::p::WRONG_CONST as RIGHT_CONST;
    pub use super::p::WrongType as RightType;
}
mod p {
    fn wrong_fn() {}
    const WRONG_CONST: u32 = 1;
    struct WrongType {};
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
    fn wrong_fn() {}
    const WRONG_CONST: u32 = 1;
    struct WrongType {};
}
"#,
        );
    }

    #[test]
    fn completes_in_simple_macro_call() {
        check(
            r#"
macro_rules! m { ($e:expr) => { $e } }
fn main() { m!(self::f<|>); }
fn foo() {}
"#,
            expect![[r#"
                fn foo()  fn foo()
                fn main() fn main()
            "#]],
        );
    }

    #[test]
    fn function_mod_share_name() {
        check(
            r#"
fn foo() { self::m::<|> }

mod m {
    pub mod z {}
    pub fn z() {}
}
"#,
            expect![[r#"
                md z
                fn z() pub fn z()
            "#]],
        );
    }

    #[test]
    fn completes_hashmap_new() {
        check(
            r#"
struct RandomState;
struct HashMap<K, V, S = RandomState> {}

impl<K, V> HashMap<K, V, RandomState> {
    pub fn new() -> HashMap<K, V, RandomState> { }
}
fn foo() {
    HashMap::<|>
}
"#,
            expect![[r#"
                fn new() pub fn new() -> HashMap<K, V, RandomState>
            "#]],
        );
    }

    #[test]
    fn dont_complete_attr() {
        check(
            r#"
mod foo { pub struct Foo; }
#[foo::<|>]
fn f() {}
"#,
            expect![[""]],
        );
    }
}
