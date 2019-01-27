use join_to_string::join;

use hir::{Docs, Resolution};

use crate::{
    completion::{CompletionItem, CompletionItemKind, Completions, CompletionKind, CompletionContext},
};

pub(super) fn complete_path(acc: &mut Completions, ctx: &CompletionContext) {
    let path = match &ctx.path_prefix {
        Some(path) => path.clone(),
        _ => return,
    };
    let def = match ctx.resolver.resolve_path(ctx.db, &path).take_types() {
        Some(Resolution::Def { def }) => def,
        _ => return,
    };
    match def {
        hir::ModuleDef::Module(module) => {
            let module_scope = module.scope(ctx.db);
            for (name, res) in module_scope.entries() {
                CompletionItem::new(
                    CompletionKind::Reference,
                    ctx.source_range(),
                    name.to_string(),
                )
                .from_resolution(ctx, &res.def.map(|def| hir::Resolution::Def { def }))
                .add_to(acc);
            }
        }
        hir::ModuleDef::Enum(e) => {
            e.variants(ctx.db).into_iter().for_each(|variant| {
                if let Some(name) = variant.name(ctx.db) {
                    let detail_types = variant
                        .fields(ctx.db)
                        .into_iter()
                        .map(|field| field.ty(ctx.db));
                    let detail = join(detail_types)
                        .separator(", ")
                        .surround_with("(", ")")
                        .to_string();

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
        _ => return,
    };
}

#[cfg(test)]
mod tests {
    use crate::completion::CompletionKind;
    use crate::completion::completion_item::check_completion;

    fn check_reference_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Reference);
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
}
