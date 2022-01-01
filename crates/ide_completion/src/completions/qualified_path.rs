//! Completion of paths, i.e. `some::prefix::$0`.

use std::iter;

use hir::{ScopeDef, Trait};
use rustc_hash::FxHashSet;
use syntax::{ast, AstNode};

use crate::{
    context::{PathCompletionContext, PathKind},
    patterns::ImmediateLocation,
    CompletionContext, Completions,
};

pub(crate) fn complete_qualified_path(acc: &mut Completions, ctx: &CompletionContext) {
    if ctx.is_path_disallowed() || ctx.has_impl_or_trait_prev_sibling() {
        return;
    }
    let (path, use_tree_parent, kind) = match ctx.path_context {
        // let ... else, syntax would come in really handy here right now
        Some(PathCompletionContext {
            qualifier: Some(ref qualifier),
            use_tree_parent,
            kind,
            ..
        }) => (qualifier, use_tree_parent, kind),
        _ => return,
    };

    // special case `<_>::$0` as this doesn't resolve to anything.
    if path.qualifier().is_none() {
        if matches!(
            path.segment().and_then(|it| it.kind()),
            Some(ast::PathSegmentKind::Type {
                type_ref: Some(ast::Type::InferType(_)),
                trait_ref: None,
            })
        ) {
            cov_mark::hit!(completion_type_anchor_empty);
            ctx.scope
                .visible_traits()
                .into_iter()
                .flat_map(|it| Trait::from(it).items(ctx.sema.db))
                .for_each(|item| add_assoc_item(acc, ctx, item));
            return;
        }
    }

    let resolution = match ctx.sema.resolve_path(path) {
        Some(res) => res,
        None => return,
    };

    let context_module = ctx.scope.module();

    match ctx.completion_location {
        Some(ImmediateLocation::ItemList | ImmediateLocation::Trait | ImmediateLocation::Impl) => {
            if let hir::PathResolution::Def(hir::ModuleDef::Module(module)) = resolution {
                for (name, def) in module.scope(ctx.db, context_module) {
                    if let ScopeDef::MacroDef(macro_def) = def {
                        if macro_def.is_fn_like() {
                            acc.add_macro(ctx, Some(name.clone()), macro_def);
                        }
                    }
                    if let ScopeDef::ModuleDef(hir::ModuleDef::Module(_)) = def {
                        acc.add_resolution(ctx, name, def);
                    }
                }
            }
            return;
        }
        _ => (),
    }

    match kind {
        Some(PathKind::Vis { .. }) => {
            if let hir::PathResolution::Def(hir::ModuleDef::Module(module)) = resolution {
                if let Some(current_module) = ctx.scope.module() {
                    if let Some(next) = current_module
                        .path_to_root(ctx.db)
                        .into_iter()
                        .take_while(|&it| it != module)
                        .next()
                    {
                        if let Some(name) = next.name(ctx.db) {
                            acc.add_resolution(ctx, name, ScopeDef::ModuleDef(next.into()));
                        }
                    }
                }
            }
            return;
        }
        Some(PathKind::Attr) => {
            if let hir::PathResolution::Def(hir::ModuleDef::Module(module)) = resolution {
                for (name, def) in module.scope(ctx.db, context_module) {
                    let add_resolution = match def {
                        ScopeDef::MacroDef(mac) => mac.is_attr(),
                        ScopeDef::ModuleDef(hir::ModuleDef::Module(_)) => true,
                        _ => false,
                    };
                    if add_resolution {
                        acc.add_resolution(ctx, name, def);
                    }
                }
            }
            return;
        }
        Some(PathKind::Use) => {
            if iter::successors(Some(path.clone()), |p| p.qualifier())
                .all(|p| p.segment().and_then(|s| s.super_token()).is_some())
            {
                acc.add_keyword(ctx, "super::");
            }
            // only show `self` in a new use-tree when the qualifier doesn't end in self
            if use_tree_parent
                && !matches!(
                    path.segment().and_then(|it| it.kind()),
                    Some(ast::PathSegmentKind::SelfKw)
                )
            {
                acc.add_keyword(ctx, "self");
            }
        }
        _ => (),
    }

    if !matches!(kind, Some(PathKind::Pat)) {
        // Add associated types on type parameters and `Self`.
        resolution.assoc_type_shorthand_candidates(ctx.db, |_, alias| {
            acc.add_type_alias(ctx, alias);
            None::<()>
        });
    }

    match resolution {
        hir::PathResolution::Def(hir::ModuleDef::Module(module)) => {
            let module_scope = module.scope(ctx.db, context_module);
            for (name, def) in module_scope {
                if let Some(PathKind::Use) = kind {
                    if let ScopeDef::Unknown = def {
                        if let Some(ast::NameLike::NameRef(name_ref)) = ctx.name_syntax.as_ref() {
                            if name_ref.syntax().text() == name.to_smol_str().as_str() {
                                // for `use self::foo$0`, don't suggest `foo` as a completion
                                cov_mark::hit!(dont_complete_current_use);
                                continue;
                            }
                        }
                    }
                }

                let add_resolution = match def {
                    // Don't suggest attribute macros and derives.
                    ScopeDef::MacroDef(mac) => mac.is_fn_like(),
                    // no values in type places
                    ScopeDef::ModuleDef(
                        hir::ModuleDef::Function(_)
                        | hir::ModuleDef::Variant(_)
                        | hir::ModuleDef::Static(_),
                    )
                    | ScopeDef::Local(_) => !ctx.expects_type(),
                    // unless its a constant in a generic arg list position
                    ScopeDef::ModuleDef(hir::ModuleDef::Const(_)) => {
                        !ctx.expects_type() || ctx.expects_generic_arg()
                    }
                    _ => true,
                };

                if add_resolution {
                    acc.add_resolution(ctx, name, def);
                }
            }
        }
        hir::PathResolution::Def(
            def
            @
            (hir::ModuleDef::Adt(_)
            | hir::ModuleDef::TypeAlias(_)
            | hir::ModuleDef::BuiltinType(_)),
        ) => {
            if let hir::ModuleDef::Adt(hir::Adt::Enum(e)) = def {
                add_enum_variants(acc, ctx, e);
            }
            let ty = match def {
                hir::ModuleDef::Adt(adt) => adt.ty(ctx.db),
                hir::ModuleDef::TypeAlias(a) => {
                    let ty = a.ty(ctx.db);
                    if let Some(hir::Adt::Enum(e)) = ty.as_adt() {
                        cov_mark::hit!(completes_variant_through_alias);
                        add_enum_variants(acc, ctx, e);
                    }
                    ty
                }
                hir::ModuleDef::BuiltinType(builtin) => {
                    let module = match ctx.scope.module() {
                        Some(it) => it,
                        None => return,
                    };
                    cov_mark::hit!(completes_primitive_assoc_const);
                    builtin.ty(ctx.db, module)
                }
                _ => unreachable!(),
            };

            // XXX: For parity with Rust bug #22519, this does not complete Ty::AssocType.
            // (where AssocType is defined on a trait, not an inherent impl)

            let krate = ctx.krate;
            if let Some(krate) = krate {
                let traits_in_scope = ctx.scope.visible_traits();
                ty.iterate_path_candidates(ctx.db, krate, &traits_in_scope, None, |_ty, item| {
                    add_assoc_item(acc, ctx, item);
                    None::<()>
                });

                // Iterate assoc types separately
                ty.iterate_assoc_items(ctx.db, krate, |item| {
                    if let hir::AssocItem::TypeAlias(ty) = item {
                        acc.add_type_alias(ctx, ty)
                    }
                    None::<()>
                });
            }
        }
        hir::PathResolution::Def(hir::ModuleDef::Trait(t)) => {
            // Handles `Trait::assoc` as well as `<Ty as Trait>::assoc`.
            for item in t.items(ctx.db) {
                add_assoc_item(acc, ctx, item);
            }
        }
        hir::PathResolution::TypeParam(_) | hir::PathResolution::SelfType(_) => {
            if let Some(krate) = ctx.krate {
                let ty = match resolution {
                    hir::PathResolution::TypeParam(param) => param.ty(ctx.db),
                    hir::PathResolution::SelfType(impl_def) => impl_def.self_ty(ctx.db),
                    _ => return,
                };

                if let Some(hir::Adt::Enum(e)) = ty.as_adt() {
                    add_enum_variants(acc, ctx, e);
                }

                let traits_in_scope = ctx.scope.visible_traits();
                let mut seen = FxHashSet::default();
                ty.iterate_path_candidates(ctx.db, krate, &traits_in_scope, None, |_ty, item| {
                    // We might iterate candidates of a trait multiple times here, so deduplicate
                    // them.
                    if seen.insert(item) {
                        add_assoc_item(acc, ctx, item);
                    }
                    None::<()>
                });
            }
        }
        hir::PathResolution::Macro(mac) => acc.add_macro(ctx, None, mac),
        _ => {}
    }
}

fn add_assoc_item(acc: &mut Completions, ctx: &CompletionContext, item: hir::AssocItem) {
    match item {
        hir::AssocItem::Function(func) if !ctx.expects_type() => acc.add_function(ctx, func, None),
        hir::AssocItem::Const(ct) if !ctx.expects_type() || ctx.expects_generic_arg() => {
            acc.add_const(ctx, ct)
        }
        hir::AssocItem::TypeAlias(ty) => acc.add_type_alias(ctx, ty),
        _ => (),
    }
}

fn add_enum_variants(acc: &mut Completions, ctx: &CompletionContext, e: hir::Enum) {
    if ctx.expects_type() {
        return;
    }
    e.variants(ctx.db).into_iter().for_each(|variant| acc.add_enum_variant(ctx, variant, None));
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::tests::{check_edit, completion_list_no_kw};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list_no_kw(ra_fixture);
        expect.assert_eq(&actual);
    }

    #[test]
    fn associated_item_visibility() {
        check(
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
                fn public_method() fn()
                ct PUBLIC_CONST    pub const PUBLIC_CONST: u32
                ta PublicType      pub type PublicType = u32
            "#]],
        );
    }

    #[test]
    fn completes_union_associated_method() {
        check(
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
        check(
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
        check(
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
        check(
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

fn foo<T: Sub>() { T::$0 }
"#,
            expect![[r#"
                ta SubTy (as Sub)        type SubTy
                ta Ty (as Super)         type Ty
                ct C2 (as Sub)           const C2: ()
                fn subfunc() (as Sub)    fn()
                me submethod(…) (as Sub) fn(&self)
                ct CONST (as Super)      const CONST: u8
                fn func() (as Super)     fn()
                me method(…) (as Super)  fn(&self)
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
        Self::$0
    }
}
"#,
            expect![[r#"
                ta SubTy (as Sub)        type SubTy
                ta Ty (as Super)         type Ty
                ct CONST (as Super)      const CONST: u8
                fn func() (as Super)     fn()
                me method(…) (as Super)  fn(&self)
                ct C2 (as Sub)           const C2: ()
                fn subfunc() (as Sub)    fn()
                me submethod(…) (as Sub) fn(&self)
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

fn main() { T::$0; }
"#,
            expect![[r#"
                fn foo() fn()
                fn bar() fn()
            "#]],
        );
    }

    #[test]
    fn completes_qualified_macros() {
        check(
            r#"
#[macro_export]
macro_rules! foo { () => {} }

fn main() { let _ = crate::$0 }
"#,
            expect![[r##"
                fn main()  fn()
                ma foo!(…) #[macro_export] macro_rules! foo
            "##]],
        );
    }

    #[test]
    fn does_not_complete_non_fn_macros() {
        check(
            r#"
mod m {
    #[rustc_builtin_macro]
    pub macro Clone {}
}

fn f() {m::$0}
"#,
            expect![[r#""#]],
        );
        check(
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
        check(
            r#"
fn foo() { self::m::$0 }

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
fn main() { m!(self::f$0); }
fn foo() {}
"#,
            expect![[r#"
                fn main() fn()
                fn foo()  fn()
            "#]],
        );
    }

    #[test]
    fn function_mod_share_name() {
        check(
            r#"
fn foo() { self::m::$0 }

mod m {
    pub mod z {}
    pub fn z() {}
}
"#,
            expect![[r#"
                md z
                fn z() fn()
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
    HashMap::$0
}
"#,
            expect![[r#"
                fn new() fn() -> HashMap<K, V, RandomState>
            "#]],
        );
    }

    #[test]
    fn dont_complete_attr() {
        check(
            r#"
mod foo { pub struct Foo; }
#[foo::$0]
fn f() {}
"#,
            expect![[""]],
        );
    }

    #[test]
    fn completes_variant_through_self() {
        check(
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
                ev Bar    ()
                ev Baz    ()
                me foo(…) fn(self)
            "#]],
        );
    }

    #[test]
    fn completes_primitive_assoc_const() {
        cov_mark::check!(completes_primitive_assoc_const);
        check(
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
        check(
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
                ev Bar ()
            "#]],
        );
    }

    #[test]
    fn respects_doc_hidden() {
        cov_mark::check!(qualified_path_doc_hidden);
        check(
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
        cov_mark::check!(completion_type_anchor_empty);
        check(
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
        )
    }
}
