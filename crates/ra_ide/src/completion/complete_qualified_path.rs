//! Completion of paths, i.e. `some::prefix::<|>`.

use hir::{Adt, HasVisibility, PathResolution, ScopeDef};
use ra_syntax::AstNode;
use rustc_hash::FxHashSet;
use test_utils::mark;

use crate::completion::{CompletionContext, Completions};

pub(super) fn complete_qualified_path(acc: &mut Completions, ctx: &CompletionContext) {
    let path = match &ctx.path_prefix {
        Some(path) => path.clone(),
        None => return,
    };

    if ctx.attribute_under_caret.is_some() {
        return;
    }

    let scope = ctx.scope();
    let context_module = scope.module();

    let res = match scope.resolve_hir_path_qualifier(&path) {
        Some(res) => res,
        None => return,
    };

    // Add associated types on type parameters and `Self`.
    res.assoc_type_shorthand_candidates(ctx.db, |alias| {
        acc.add_type_alias(ctx, alias);
        None::<()>
    });

    match res {
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
                let traits_in_scope = ctx.scope().traits_in_scope();
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
                let ty = match res {
                    PathResolution::TypeParam(param) => param.ty(ctx.db),
                    PathResolution::SelfType(impl_def) => impl_def.target_ty(ctx.db),
                    _ => return,
                };

                let traits_in_scope = ctx.scope().traits_in_scope();
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
    use test_utils::mark;

    use crate::completion::{test_utils::do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot;

    fn do_reference_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Reference)
    }

    #[test]
    fn dont_complete_current_use() {
        mark::check!(dont_complete_current_use);
        let completions = do_completion(r"use self::foo<|>;", CompletionKind::Reference);
        assert!(completions.is_empty());
    }

    #[test]
    fn dont_complete_current_use_in_braces_with_glob() {
        let completions = do_completion(
            r"
            mod foo { pub struct S; }
            use self::{foo::*, bar<|>};
            ",
            CompletionKind::Reference,
        );
        assert_eq!(completions.len(), 2);
    }

    #[test]
    fn dont_complete_primitive_in_use() {
        let completions = do_completion(r"use self::<|>;", CompletionKind::BuiltinType);
        assert!(completions.is_empty());
    }

    #[test]
    fn dont_complete_primitive_in_module_scope() {
        let completions = do_completion(r"fn foo() { self::<|> }", CompletionKind::BuiltinType);
        assert!(completions.is_empty());
    }

    #[test]
    fn completes_primitives() {
        let completions =
            do_completion(r"fn main() { let _: <|> = 92; }", CompletionKind::BuiltinType);
        assert_eq!(completions.len(), 17);
    }

    #[test]
    fn completes_mod_with_docs() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                use self::my<|>;

                /// Some simple
                /// docs describing `mod my`.
                mod my {
                    struct Bar;
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "my",
                source_range: 10..12,
                delete: 10..12,
                insert: "my",
                kind: Module,
                documentation: Documentation(
                    "Some simple\ndocs describing `mod my`.",
                ),
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_mod_with_same_name_as_function() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                use self::my::<|>;

                mod my {
                    pub struct Bar;
                }

                fn my() {}
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Bar",
                source_range: 14..14,
                delete: 14..14,
                insert: "Bar",
                kind: Struct,
            },
        ]
        "###
        );
    }

    #[test]
    fn path_visibility() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                use self::my::<|>;

                mod my {
                    struct Bar;
                    pub struct Foo;
                    pub use Bar as PublicBar;
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Foo",
                source_range: 14..14,
                delete: 14..14,
                insert: "Foo",
                kind: Struct,
            },
            CompletionItem {
                label: "PublicBar",
                source_range: 14..14,
                delete: 14..14,
                insert: "PublicBar",
                kind: Struct,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_use_item_starting_with_self() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                use self::m::<|>;

                mod m {
                    pub struct Bar;
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Bar",
                source_range: 13..13,
                delete: 13..13,
                insert: "Bar",
                kind: Struct,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_use_item_starting_with_crate() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                mod foo;
                struct Spam;
                //- /foo.rs
                use crate::Sp<|>
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Spam",
                source_range: 11..13,
                delete: 11..13,
                insert: "Spam",
                kind: Struct,
            },
            CompletionItem {
                label: "foo",
                source_range: 11..13,
                delete: 11..13,
                insert: "foo",
                kind: Module,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_nested_use_tree() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                mod foo;
                struct Spam;
                //- /foo.rs
                use crate::{Sp<|>};
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Spam",
                source_range: 12..14,
                delete: 12..14,
                insert: "Spam",
                kind: Struct,
            },
            CompletionItem {
                label: "foo",
                source_range: 12..14,
                delete: 12..14,
                insert: "foo",
                kind: Module,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_deeply_nested_use_tree() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                mod foo;
                pub mod bar {
                    pub mod baz {
                        pub struct Spam;
                    }
                }
                //- /foo.rs
                use crate::{bar::{baz::Sp<|>}};
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Spam",
                source_range: 23..25,
                delete: 23..25,
                insert: "Spam",
                kind: Struct,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_enum_variant() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                /// An enum
                enum E {
                    /// Foo Variant
                    Foo,
                    /// Bar Variant with i32
                    Bar(i32)
                }
                fn foo() { let _ = E::<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Bar(…)",
                source_range: 116..116,
                delete: 116..116,
                insert: "Bar($0)",
                kind: EnumVariant,
                lookup: "Bar",
                detail: "(i32)",
                documentation: Documentation(
                    "Bar Variant with i32",
                ),
                trigger_call_info: true,
            },
            CompletionItem {
                label: "Foo",
                source_range: 116..116,
                delete: 116..116,
                insert: "Foo",
                kind: EnumVariant,
                detail: "()",
                documentation: Documentation(
                    "Foo Variant",
                ),
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_enum_variant_with_details() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                struct S { field: u32 }
                /// An enum
                enum E {
                    /// Foo Variant (empty)
                    Foo,
                    /// Bar Variant with i32 and u32
                    Bar(i32, u32),
                    ///
                    S(S),
                }
                fn foo() { let _ = E::<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Bar(…)",
                source_range: 180..180,
                delete: 180..180,
                insert: "Bar($0)",
                kind: EnumVariant,
                lookup: "Bar",
                detail: "(i32, u32)",
                documentation: Documentation(
                    "Bar Variant with i32 and u32",
                ),
                trigger_call_info: true,
            },
            CompletionItem {
                label: "Foo",
                source_range: 180..180,
                delete: 180..180,
                insert: "Foo",
                kind: EnumVariant,
                detail: "()",
                documentation: Documentation(
                    "Foo Variant (empty)",
                ),
            },
            CompletionItem {
                label: "S(…)",
                source_range: 180..180,
                delete: 180..180,
                insert: "S($0)",
                kind: EnumVariant,
                lookup: "S",
                detail: "(S)",
                documentation: Documentation(
                    "",
                ),
                trigger_call_info: true,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_struct_associated_method() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                /// A Struct
                struct S;

                impl S {
                    /// An associated method
                    fn m() { }
                }

                fn foo() { let _ = S::<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "m()",
                source_range: 102..102,
                delete: 102..102,
                insert: "m()$0",
                kind: Function,
                lookup: "m",
                detail: "fn m()",
                documentation: Documentation(
                    "An associated method",
                ),
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_struct_associated_method_with_self() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                /// A Struct
                struct S;

                impl S {
                    /// An associated method
                    fn m(&self) { }
                }

                fn foo() { let _ = S::<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "m()",
                source_range: 107..107,
                delete: 107..107,
                insert: "m()$0",
                kind: Method,
                lookup: "m",
                detail: "fn m(&self)",
                documentation: Documentation(
                    "An associated method",
                ),
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_struct_associated_const() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                /// A Struct
                struct S;

                impl S {
                    /// An associated const
                    const C: i32 = 42;
                }

                fn foo() { let _ = S::<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "C",
                source_range: 109..109,
                delete: 109..109,
                insert: "C",
                kind: Const,
                detail: "const C: i32 = 42;",
                documentation: Documentation(
                    "An associated const",
                ),
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_struct_associated_type() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                /// A Struct
                struct S;

                impl S {
                    /// An associated type
                    type T = i32;
                }

                fn foo() { let _ = S::<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "T",
                source_range: 103..103,
                delete: 103..103,
                insert: "T",
                kind: TypeAlias,
                detail: "type T = i32;",
                documentation: Documentation(
                    "An associated type",
                ),
            },
        ]
        "###
        );
    }

    #[test]
    fn associated_item_visibility() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
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
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "PUBLIC_CONST",
                source_range: 304..304,
                delete: 304..304,
                insert: "PUBLIC_CONST",
                kind: Const,
                detail: "pub(super) const PUBLIC_CONST: u32 = 1;",
            },
            CompletionItem {
                label: "PublicType",
                source_range: 304..304,
                delete: 304..304,
                insert: "PublicType",
                kind: TypeAlias,
                detail: "pub(super) type PublicType = u32;",
            },
            CompletionItem {
                label: "public_method()",
                source_range: 304..304,
                delete: 304..304,
                insert: "public_method()$0",
                kind: Function,
                lookup: "public_method",
                detail: "pub(super) fn public_method()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_enum_associated_method() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                /// An enum
                enum S {};

                impl S {
                    /// An associated method
                    fn m() { }
                }

                fn foo() { let _ = S::<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "m()",
                source_range: 102..102,
                delete: 102..102,
                insert: "m()$0",
                kind: Function,
                lookup: "m",
                detail: "fn m()",
                documentation: Documentation(
                    "An associated method",
                ),
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_union_associated_method() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                /// A union
                union U {};

                impl U {
                    /// An associated method
                    fn m() { }
                }

                fn foo() { let _ = U::<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "m()",
                source_range: 103..103,
                delete: 103..103,
                insert: "m()$0",
                kind: Function,
                lookup: "m",
                detail: "fn m()",
                documentation: Documentation(
                    "An associated method",
                ),
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_use_paths_across_crates() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /main.rs
                use foo::<|>;

                //- /foo/lib.rs
                pub mod bar {
                    pub struct S;
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "bar",
                source_range: 9..9,
                delete: 9..9,
                insert: "bar",
                kind: Module,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_trait_associated_method_1() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                trait Trait {
                  /// A trait method
                  fn m();
                }

                fn foo() { let _ = Trait::<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "m()",
                source_range: 74..74,
                delete: 74..74,
                insert: "m()$0",
                kind: Function,
                lookup: "m",
                detail: "fn m()",
                documentation: Documentation(
                    "A trait method",
                ),
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_trait_associated_method_2() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                trait Trait {
                  /// A trait method
                  fn m();
                }

                struct S;
                impl Trait for S {}

                fn foo() { let _ = S::<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "m()",
                source_range: 101..101,
                delete: 101..101,
                insert: "m()$0",
                kind: Function,
                lookup: "m",
                detail: "fn m()",
                documentation: Documentation(
                    "A trait method",
                ),
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_trait_associated_method_3() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                trait Trait {
                  /// A trait method
                  fn m();
                }

                struct S;
                impl Trait for S {}

                fn foo() { let _ = <S as Trait>::<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "m()",
                source_range: 112..112,
                delete: 112..112,
                insert: "m()$0",
                kind: Function,
                lookup: "m",
                detail: "fn m()",
                documentation: Documentation(
                    "A trait method",
                ),
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_ty_param_assoc_ty() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
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

                fn foo<T: Sub>() {
                    T::<|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "C2",
                source_range: 221..221,
                delete: 221..221,
                insert: "C2",
                kind: Const,
                detail: "const C2: ();",
            },
            CompletionItem {
                label: "CONST",
                source_range: 221..221,
                delete: 221..221,
                insert: "CONST",
                kind: Const,
                detail: "const CONST: u8;",
            },
            CompletionItem {
                label: "SubTy",
                source_range: 221..221,
                delete: 221..221,
                insert: "SubTy",
                kind: TypeAlias,
                detail: "type SubTy;",
            },
            CompletionItem {
                label: "Ty",
                source_range: 221..221,
                delete: 221..221,
                insert: "Ty",
                kind: TypeAlias,
                detail: "type Ty;",
            },
            CompletionItem {
                label: "func()",
                source_range: 221..221,
                delete: 221..221,
                insert: "func()$0",
                kind: Function,
                lookup: "func",
                detail: "fn func()",
            },
            CompletionItem {
                label: "method()",
                source_range: 221..221,
                delete: 221..221,
                insert: "method()$0",
                kind: Method,
                lookup: "method",
                detail: "fn method(&self)",
            },
            CompletionItem {
                label: "subfunc()",
                source_range: 221..221,
                delete: 221..221,
                insert: "subfunc()$0",
                kind: Function,
                lookup: "subfunc",
                detail: "fn subfunc()",
            },
            CompletionItem {
                label: "submethod()",
                source_range: 221..221,
                delete: 221..221,
                insert: "submethod()$0",
                kind: Method,
                lookup: "submethod",
                detail: "fn submethod(&self)",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_self_param_assoc_ty() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
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
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "C2",
                source_range: 367..367,
                delete: 367..367,
                insert: "C2",
                kind: Const,
                detail: "const C2: () = ();",
            },
            CompletionItem {
                label: "CONST",
                source_range: 367..367,
                delete: 367..367,
                insert: "CONST",
                kind: Const,
                detail: "const CONST: u8 = 0;",
            },
            CompletionItem {
                label: "SubTy",
                source_range: 367..367,
                delete: 367..367,
                insert: "SubTy",
                kind: TypeAlias,
                detail: "type SubTy;",
            },
            CompletionItem {
                label: "Ty",
                source_range: 367..367,
                delete: 367..367,
                insert: "Ty",
                kind: TypeAlias,
                detail: "type Ty;",
            },
            CompletionItem {
                label: "func()",
                source_range: 367..367,
                delete: 367..367,
                insert: "func()$0",
                kind: Function,
                lookup: "func",
                detail: "fn func()",
            },
            CompletionItem {
                label: "method()",
                source_range: 367..367,
                delete: 367..367,
                insert: "method()$0",
                kind: Method,
                lookup: "method",
                detail: "fn method(&self)",
            },
            CompletionItem {
                label: "subfunc()",
                source_range: 367..367,
                delete: 367..367,
                insert: "subfunc()$0",
                kind: Function,
                lookup: "subfunc",
                detail: "fn subfunc()",
            },
            CompletionItem {
                label: "submethod()",
                source_range: 367..367,
                delete: 367..367,
                insert: "submethod()$0",
                kind: Method,
                lookup: "submethod",
                detail: "fn submethod(&self)",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_type_alias() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                struct S;
                impl S { fn foo() {} }
                type T = S;
                impl T { fn bar() {} }

                fn main() {
                    T::<|>;
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "bar()",
                source_range: 88..88,
                delete: 88..88,
                insert: "bar()$0",
                kind: Function,
                lookup: "bar",
                detail: "fn bar()",
            },
            CompletionItem {
                label: "foo()",
                source_range: 88..88,
                delete: 88..88,
                insert: "foo()$0",
                kind: Function,
                lookup: "foo",
                detail: "fn foo()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_qualified_macros() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                #[macro_export]
                macro_rules! foo {
                    () => {}
                }

                fn main() {
                    let _ = crate::<|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "foo!(…)",
                source_range: 82..82,
                delete: 82..82,
                insert: "foo!($0)",
                kind: Macro,
                detail: "#[macro_export]\nmacro_rules! foo",
            },
            CompletionItem {
                label: "main()",
                source_range: 82..82,
                delete: 82..82,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_super_super_completion() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                mod a {
                    const A: usize = 0;

                    mod b {
                        const B: usize = 0;

                        mod c {
                            use super::super::<|>
                        }
                    }
                }
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "A",
                source_range: 120..120,
                delete: 120..120,
                insert: "A",
                kind: Const,
            },
            CompletionItem {
                label: "b",
                source_range: 120..120,
                delete: 120..120,
                insert: "b",
                kind: Module,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_reexported_items_under_correct_name() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                fn foo() {
                    self::m::<|>
                }

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
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "RIGHT_CONST",
                source_range: 24..24,
                delete: 24..24,
                insert: "RIGHT_CONST",
                kind: Const,
            },
            CompletionItem {
                label: "RightType",
                source_range: 24..24,
                delete: 24..24,
                insert: "RightType",
                kind: Struct,
            },
            CompletionItem {
                label: "right_fn()",
                source_range: 24..24,
                delete: 24..24,
                insert: "right_fn()$0",
                kind: Function,
                lookup: "right_fn",
                detail: "fn wrong_fn()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_in_simple_macro_call() {
        let completions = do_reference_completion(
            r#"
                macro_rules! m { ($e:expr) => { $e } }
                fn main() { m!(self::f<|>); }
                fn foo() {}
            "#,
        );
        assert_debug_snapshot!(completions, @r###"
        [
            CompletionItem {
                label: "foo()",
                source_range: 60..61,
                delete: 60..61,
                insert: "foo()$0",
                kind: Function,
                lookup: "foo",
                detail: "fn foo()",
            },
            CompletionItem {
                label: "main()",
                source_range: 60..61,
                delete: 60..61,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
        ]
        "###);
    }

    #[test]
    fn function_mod_share_name() {
        assert_debug_snapshot!(
        do_reference_completion(
                r"
                fn foo() {
                    self::m::<|>
                }

                mod m {
                    pub mod z {}
                    pub fn z() {}
                }
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "z",
                source_range: 24..24,
                delete: 24..24,
                insert: "z",
                kind: Module,
            },
            CompletionItem {
                label: "z()",
                source_range: 24..24,
                delete: 24..24,
                insert: "z()$0",
                kind: Function,
                lookup: "z",
                detail: "pub fn z()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_hashmap_new() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                struct RandomState;
                struct HashMap<K, V, S = RandomState> {}

                impl<K, V> HashMap<K, V, RandomState> {
                    pub fn new() -> HashMap<K, V, RandomState> { }
                }
                fn foo() {
                    HashMap::<|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "new()",
                source_range: 179..179,
                delete: 179..179,
                insert: "new()$0",
                kind: Function,
                lookup: "new",
                detail: "pub fn new() -> HashMap<K, V, RandomState>",
            },
        ]
        "###
        );
    }

    #[test]
    fn dont_complete_attr() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                mod foo { pub struct Foo; }
                #[foo::<|>]
                fn f() {}
                "
            ),
            @r###"[]"###
        )
    }
}
