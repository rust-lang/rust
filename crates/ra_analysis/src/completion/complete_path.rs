use crate::{
    Cancelable,
    completion::{CompletionItem, Completions, CompletionKind, CompletionContext},
};

pub(super) fn complete_path(acc: &mut Completions, ctx: &CompletionContext) -> Cancelable<()> {
    let (path, module) = match (&ctx.path_prefix, &ctx.module) {
        (Some(path), Some(module)) => (path.clone(), module),
        _ => return Ok(()),
    };
    let def_id = match module.resolve_path(ctx.db, path)? {
        Some(it) => it,
        None => return Ok(()),
    };
    let target_module = match def_id.resolve(ctx.db)? {
        hir::Def::Module(it) => it,
        _ => return Ok(()),
    };
    let module_scope = target_module.scope(ctx.db)?;
    module_scope.entries().for_each(|(name, _res)| {
        CompletionItem::new(CompletionKind::Reference, name.to_string()).add_to(acc)
    });
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
}
