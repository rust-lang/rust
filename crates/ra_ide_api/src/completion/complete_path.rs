//! FIXME: write short doc here

use hir::{Adt, Either, PathResolution};
use ra_syntax::AstNode;
use test_utils::tested_by;

use crate::completion::{CompletionContext, Completions};

pub(super) fn complete_path(acc: &mut Completions, ctx: &CompletionContext) {
    let path = match &ctx.path_prefix {
        Some(path) => path.clone(),
        _ => return,
    };
    let def = match ctx.analyzer.resolve_hir_path(ctx.db, &path) {
        Some(PathResolution::Def(def)) => def,
        _ => return,
    };
    match def {
        hir::ModuleDef::Module(module) => {
            let module_scope = module.scope(ctx.db);
            for (name, def, import) in module_scope {
                if let hir::ScopeDef::ModuleDef(hir::ModuleDef::BuiltinType(..)) = def {
                    if ctx.use_item_syntax.is_some() {
                        tested_by!(dont_complete_primitive_in_use);
                        continue;
                    }
                }
                if Some(module) == ctx.module {
                    if let Some(import) = import {
                        if let Either::A(use_tree) = module.import_source(ctx.db, import) {
                            if use_tree.syntax().text_range().contains_inclusive(ctx.offset) {
                                // for `use self::foo<|>`, don't suggest `foo` as a completion
                                tested_by!(dont_complete_current_use);
                                continue;
                            }
                        }
                    }
                }
                acc.add_resolution(ctx, name.to_string(), &def);
            }
        }
        hir::ModuleDef::Adt(_) | hir::ModuleDef::TypeAlias(_) => {
            if let hir::ModuleDef::Adt(Adt::Enum(e)) = def {
                for variant in e.variants(ctx.db) {
                    acc.add_enum_variant(ctx, variant);
                }
            }
            let ty = match def {
                hir::ModuleDef::Adt(adt) => adt.ty(ctx.db),
                hir::ModuleDef::TypeAlias(a) => a.ty(ctx.db),
                _ => unreachable!(),
            };
            ctx.analyzer.iterate_path_candidates(ctx.db, ty.clone(), None, |_ty, item| {
                match item {
                    hir::AssocItem::Function(func) => {
                        let data = func.data(ctx.db);
                        if !data.has_self_param() {
                            acc.add_function(ctx, func);
                        }
                    }
                    hir::AssocItem::Const(ct) => acc.add_const(ctx, ct),
                    hir::AssocItem::TypeAlias(ty) => acc.add_type_alias(ctx, ty),
                }
                None::<()>
            });
            // Iterate assoc types separately
            // FIXME: complete T::AssocType
            let krate = ctx.module.map(|m| m.krate());
            if let Some(krate) = krate {
                ty.iterate_impl_items(ctx.db, krate, |item| {
                    match item {
                        hir::AssocItem::Function(_) | hir::AssocItem::Const(_) => {}
                        hir::AssocItem::TypeAlias(ty) => acc.add_type_alias(ctx, ty),
                    }
                    None::<()>
                });
            }
        }
        hir::ModuleDef::Trait(t) => {
            for item in t.items(ctx.db) {
                match item {
                    hir::AssocItem::Function(func) => {
                        let data = func.data(ctx.db);
                        if !data.has_self_param() {
                            acc.add_function(ctx, func);
                        }
                    }
                    hir::AssocItem::Const(ct) => acc.add_const(ctx, ct),
                    hir::AssocItem::TypeAlias(ty) => acc.add_type_alias(ctx, ty),
                }
            }
        }
        _ => {}
    };
}

#[cfg(test)]
mod tests {
    use test_utils::covers;

    use crate::completion::{do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot;

    fn do_reference_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Reference)
    }

    #[test]
    fn dont_complete_current_use() {
        covers!(dont_complete_current_use);
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
        covers!(dont_complete_primitive_in_use);
        let completions = do_completion(r"use self::<|>;", CompletionKind::BuiltinType);
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
            @r###"[
    CompletionItem {
        label: "my",
        source_range: [27; 29),
        delete: [27; 29),
        insert: "my",
        kind: Module,
        documentation: Documentation(
            "Some simple\ndocs describing `mod my`.",
        ),
    },
]"###
        );
    }

    #[test]
    fn completes_use_item_starting_with_self() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                use self::m::<|>;

                mod m {
                    struct Bar;
                }
                "
            ),
            @r###"[
    CompletionItem {
        label: "Bar",
        source_range: [30; 30),
        delete: [30; 30),
        insert: "Bar",
        kind: Struct,
    },
]"###
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
            @r###"[
    CompletionItem {
        label: "Spam",
        source_range: [11; 13),
        delete: [11; 13),
        insert: "Spam",
        kind: Struct,
    },
    CompletionItem {
        label: "foo",
        source_range: [11; 13),
        delete: [11; 13),
        insert: "foo",
        kind: Module,
    },
]"###
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
            @r###"[
    CompletionItem {
        label: "Spam",
        source_range: [12; 14),
        delete: [12; 14),
        insert: "Spam",
        kind: Struct,
    },
    CompletionItem {
        label: "foo",
        source_range: [12; 14),
        delete: [12; 14),
        insert: "foo",
        kind: Module,
    },
]"###
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
            @r###"[
    CompletionItem {
        label: "Spam",
        source_range: [23; 25),
        delete: [23; 25),
        insert: "Spam",
        kind: Struct,
    },
]"###
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
            @r###"[
    CompletionItem {
        label: "Bar",
        source_range: [116; 116),
        delete: [116; 116),
        insert: "Bar",
        kind: EnumVariant,
        detail: "(i32)",
        documentation: Documentation(
            "Bar Variant with i32",
        ),
    },
    CompletionItem {
        label: "Foo",
        source_range: [116; 116),
        delete: [116; 116),
        insert: "Foo",
        kind: EnumVariant,
        detail: "()",
        documentation: Documentation(
            "Foo Variant",
        ),
    },
]"###
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
            @r###"[
    CompletionItem {
        label: "Bar",
        source_range: [180; 180),
        delete: [180; 180),
        insert: "Bar",
        kind: EnumVariant,
        detail: "(i32, u32)",
        documentation: Documentation(
            "Bar Variant with i32 and u32",
        ),
    },
    CompletionItem {
        label: "Foo",
        source_range: [180; 180),
        delete: [180; 180),
        insert: "Foo",
        kind: EnumVariant,
        detail: "()",
        documentation: Documentation(
            "Foo Variant (empty)",
        ),
    },
    CompletionItem {
        label: "S",
        source_range: [180; 180),
        delete: [180; 180),
        insert: "S",
        kind: EnumVariant,
        detail: "(S)",
        documentation: Documentation(
            "",
        ),
    },
]"###
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
                source_range: [100; 100),
                delete: [100; 100),
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
            @r###"[
    CompletionItem {
        label: "C",
        source_range: [107; 107),
        delete: [107; 107),
        insert: "C",
        kind: Const,
        detail: "const C: i32 = 42;",
        documentation: Documentation(
            "An associated const",
        ),
    },
]"###
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
            @r###"[
    CompletionItem {
        label: "T",
        source_range: [101; 101),
        delete: [101; 101),
        insert: "T",
        kind: TypeAlias,
        detail: "type T = i32;",
        documentation: Documentation(
            "An associated type",
        ),
    },
]"###
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
                source_range: [100; 100),
                delete: [100; 100),
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
                source_range: [101; 101),
                delete: [101; 101),
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
            @r###"[
    CompletionItem {
        label: "bar",
        source_range: [9; 9),
        delete: [9; 9),
        insert: "bar",
        kind: Module,
    },
]"###
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
                source_range: [73; 73),
                delete: [73; 73),
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
                source_range: [99; 99),
                delete: [99; 99),
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
                source_range: [110; 110),
                delete: [110; 110),
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
                source_range: [185; 185),
                delete: [185; 185),
                insert: "bar()$0",
                kind: Function,
                lookup: "bar",
                detail: "fn bar()",
            },
            CompletionItem {
                label: "foo()",
                source_range: [185; 185),
                delete: [185; 185),
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
                label: "foo!",
                source_range: [179; 179),
                delete: [179; 179),
                insert: "foo!($0)",
                kind: Macro,
                detail: "#[macro_export]\nmacro_rules! foo",
            },
            CompletionItem {
                label: "main()",
                source_range: [179; 179),
                delete: [179; 179),
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
        ]
        "###
        );
    }
}
