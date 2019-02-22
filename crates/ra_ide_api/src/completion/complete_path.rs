use join_to_string::join;
use hir::{Docs, Resolution};
use ra_syntax::{AstNode, ast::NameOwner};
use test_utils::tested_by;

use crate::completion::{CompletionItem, CompletionItemKind, Completions, CompletionKind, CompletionContext};

pub(super) fn complete_path(acc: &mut Completions, ctx: &CompletionContext) {
    let path = match &ctx.path_prefix {
        Some(path) => path.clone(),
        _ => return,
    };
    let def = match ctx.resolver.resolve_path(ctx.db, &path).take_types() {
        Some(Resolution::Def(def)) => def,
        _ => return,
    };
    match def {
        hir::ModuleDef::Module(module) => {
            let module_scope = module.scope(ctx.db);
            for (name, res) in module_scope.entries() {
                if Some(module) == ctx.module {
                    if let Some(import) = res.import {
                        let path = module.import_source(ctx.db, import);
                        if path.syntax().range().contains_inclusive(ctx.offset) {
                            // for `use self::foo<|>`, don't suggest `foo` as a completion
                            tested_by!(dont_complete_current_use);
                            continue;
                        }
                    }
                }

                CompletionItem::new(
                    CompletionKind::Reference,
                    ctx.source_range(),
                    name.to_string(),
                )
                .from_resolution(ctx, &res.def.map(hir::Resolution::Def))
                .add_to(acc);
            }
        }
        hir::ModuleDef::Enum(e) => {
            e.variants(ctx.db).into_iter().for_each(|variant| {
                if let Some(name) = variant.name(ctx.db) {
                    let detail_types =
                        variant.fields(ctx.db).into_iter().map(|field| field.ty(ctx.db));
                    let detail =
                        join(detail_types).separator(", ").surround_with("(", ")").to_string();

                    CompletionItem::new(
                        CompletionKind::Reference,
                        ctx.source_range(),
                        name.to_string(),
                    )
                    .kind(CompletionItemKind::EnumVariant)
                    .set_documentation(variant.docs(ctx.db))
                    .set_detail(Some(detail))
                    .add_to(acc)
                }
            });
        }
        hir::ModuleDef::Struct(s) => {
            let ty = s.ty(ctx.db);
            ty.iterate_impl_items(ctx.db, |item| match item {
                hir::ImplItem::Method(func) => {
                    let sig = func.signature(ctx.db);
                    if !sig.has_self_param() {
                        CompletionItem::new(
                            CompletionKind::Reference,
                            ctx.source_range(),
                            sig.name().to_string(),
                        )
                        .from_function(ctx, func)
                        .kind(CompletionItemKind::Method)
                        .add_to(acc);
                    }
                    None::<()>
                }
                hir::ImplItem::Const(ct) => {
                    let source = ct.source(ctx.db);
                    if let Some(name) = source.1.name() {
                        CompletionItem::new(
                            CompletionKind::Reference,
                            ctx.source_range(),
                            name.text().to_string(),
                        )
                        .from_const(ctx, ct)
                        .add_to(acc);
                    }
                    None::<()>
                }
                hir::ImplItem::Type(ty) => {
                    let source = ty.source(ctx.db);
                    if let Some(name) = source.1.name() {
                        CompletionItem::new(
                            CompletionKind::Reference,
                            ctx.source_range(),
                            name.text().to_string(),
                        )
                        .from_type(ctx, ty)
                        .add_to(acc);
                    }
                    None::<()>
                }
            });
        }
        _ => return,
    };
}

#[cfg(test)]
mod tests {
    use crate::completion::{
        CompletionKind,
        completion_item::{check_completion, do_completion},
};

    use test_utils::covers;

    fn check_reference_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Reference);
    }

    #[test]
    fn dont_complete_current_use() {
        covers!(dont_complete_current_use);
        let completions = do_completion(r"use self::foo<|>;", CompletionKind::Reference);
        assert!(completions.is_empty());
    }

    #[test]
    fn completes_mod_with_docs() {
        check_reference_completion(
            "mod_with_docs",
            r"
            use self::my<|>;

            /// Some simple
            /// docs describing `mod my`.
            mod my {
                struct Bar;
            }
            ",
        );
    }

    #[test]
    fn completes_use_item_starting_with_self() {
        check_reference_completion(
            "use_item_starting_with_self",
            r"
            use self::m::<|>;

            mod m {
                struct Bar;
            }
            ",
        );
    }

    #[test]
    fn completes_use_item_starting_with_crate() {
        check_reference_completion(
            "use_item_starting_with_crate",
            "
            //- /lib.rs
            mod foo;
            struct Spam;
            //- /foo.rs
            use crate::Sp<|>
            ",
        );
    }

    #[test]
    fn completes_nested_use_tree() {
        check_reference_completion(
            "nested_use_tree",
            "
            //- /lib.rs
            mod foo;
            struct Spam;
            //- /foo.rs
            use crate::{Sp<|>};
            ",
        );
    }

    #[test]
    fn completes_deeply_nested_use_tree() {
        check_reference_completion(
            "deeply_nested_use_tree",
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
            ",
        );
    }

    #[test]
    fn completes_enum_variant() {
        check_reference_completion(
            "enum_variant",
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
            ",
        );
    }

    #[test]
    fn completes_enum_variant_with_details() {
        check_reference_completion(
            "enum_variant_with_details",
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
            ",
        );
    }

    #[test]
    fn completes_struct_associated_method() {
        check_reference_completion(
            "struct_associated_method",
            "
            //- /lib.rs
            /// A Struct
            struct S;

            impl S {
                /// An associated method
                fn m() { }
            }

            fn foo() { let _ = S::<|> }
            ",
        );
    }

    #[test]
    fn completes_struct_associated_const() {
        check_reference_completion(
            "struct_associated_const",
            "
            //- /lib.rs
            /// A Struct
            struct S;

            impl S {
                /// An associated const
                const C: i32 = 42;
            }

            fn foo() { let _ = S::<|> }
            ",
        );
    }

    #[test]
    fn completes_struct_associated_type() {
        check_reference_completion(
            "struct_associated_type",
            "
            //- /lib.rs
            /// A Struct
            struct S;

            impl S {
                /// An associated type
                type T = i32;
            }

            fn foo() { let _ = S::<|> }
            ",
        );
    }

    #[test]
    fn completes_use_paths_across_crates() {
        check_reference_completion(
            "completes_use_paths_across_crates",
            "
            //- /main.rs
            use foo::<|>;

            //- /foo/lib.rs
            pub mod bar {
                pub struct S;
            }
            ",
        );
    }
}
