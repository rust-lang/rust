use crate::{
    completion::{CompletionItem, CompletionItemKind, Completions, CompletionKind, CompletionContext},
};

pub(super) fn complete_path(acc: &mut Completions, ctx: &CompletionContext) {
    let (path, module) = match (&ctx.path_prefix, &ctx.module) {
        (Some(path), Some(module)) => (path.clone(), module),
        _ => return,
    };
    let def_id = match module.resolve_path(ctx.db, &path).take_types() {
        Some(it) => it,
        None => return,
    };
    match def_id.resolve(ctx.db) {
        hir::Def::Module(module) => {
            let module_scope = module.scope(ctx.db);
            for (name, res) in module_scope.entries() {
                CompletionItem::new(
                    CompletionKind::Reference,
                    ctx.source_range(),
                    name.to_string(),
                )
                .from_resolution(ctx, res)
                .add_to(acc);
            }
        }
        hir::Def::Enum(e) => {
            e.variants(ctx.db)
                .into_iter()
                .for_each(|(variant_name, _variant)| {
                    CompletionItem::new(
                        CompletionKind::Reference,
                        ctx.source_range(),
                        variant_name.to_string(),
                    )
                    .kind(CompletionItemKind::EnumVariant)
                    .add_to(acc)
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
            "reference_completion",
            "
            //- /lib.rs
            enum E { Foo, Bar(i32) }
            fn foo() { let _ = E::<|> }
            ",
        );
    }
}
