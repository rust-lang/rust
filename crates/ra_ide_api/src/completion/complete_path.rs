use crate::{
    Cancelable,
    completion::{CompletionItem, CompletionItemKind, Completions, CompletionKind, CompletionContext},
};

pub(super) fn complete_path(acc: &mut Completions, ctx: &CompletionContext) -> Cancelable<()> {
    let (path, module) = match (&ctx.path_prefix, &ctx.module) {
        (Some(path), Some(module)) => (path.clone(), module),
        _ => return Ok(()),
    };
    let def_id = match module.resolve_path(ctx.db, &path)?.take_types() {
        Some(it) => it,
        None => return Ok(()),
    };
    match def_id.resolve(ctx.db)? {
        hir::Def::Module(module) => {
            let module_scope = module.scope(ctx.db)?;
            module_scope.entries().for_each(|(name, res)| {
                CompletionItem::new(CompletionKind::Reference, name.to_string())
                    .from_resolution(ctx, res)
                    .add_to(acc)
            });
        }
        hir::Def::Enum(e) => {
            e.variants(ctx.db)?
                .into_iter()
                .for_each(|(variant_name, _variant)| {
                    CompletionItem::new(CompletionKind::Reference, variant_name.to_string())
                        .kind(CompletionItemKind::EnumVariant)
                        .add_to(acc)
                });
        }
        _ => return Ok(()),
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::completion::{CompletionKind, check_completion};

    fn check_reference_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Reference);
    }

    #[test]
    fn completes_use_item_starting_with_self() {
        check_reference_completion(
            r"
            use self::m::<|>;

            mod m {
                struct Bar;
            }
            ",
            "Bar",
        );
    }

    #[test]
    fn completes_use_item_starting_with_crate() {
        check_reference_completion(
            "
            //- /lib.rs
            mod foo;
            struct Spam;
            //- /foo.rs
            use crate::Sp<|>
            ",
            "Spam;foo",
        );
    }

    #[test]
    fn completes_nested_use_tree() {
        check_reference_completion(
            "
            //- /lib.rs
            mod foo;
            struct Spam;
            //- /foo.rs
            use crate::{Sp<|>};
            ",
            "Spam;foo",
        );
    }

    #[test]
    fn completes_deeply_nested_use_tree() {
        check_reference_completion(
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
            "Spam",
        );
    }

    #[test]
    fn completes_enum_variant() {
        check_reference_completion(
            "
            //- /lib.rs
            enum E { Foo, Bar(i32) }
            fn foo() { let _ = E::<|> }
            ",
            "Foo;Bar",
        );
    }

    #[test]
    fn dont_render_function_parens_in_use_item() {
        check_reference_completion(
            "
            //- /lib.rs
            mod m { pub fn foo() {} }
            use crate::m::f<|>;
            ",
            "foo",
        )
    }
}
