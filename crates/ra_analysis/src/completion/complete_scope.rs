use rustc_hash::FxHashSet;
use ra_syntax::TextUnit;

use crate::{
    Cancelable,
    completion::{CompletionItem, Completions, CompletionKind::*, CompletionContext},
};

pub(super) fn complete_scope(acc: &mut Completions, ctx: &CompletionContext) -> Cancelable<()> {
    if !ctx.is_trivial_path {
        return Ok(());
    }
    if let Some(fn_def) = ctx.enclosing_fn {
        let scopes = hir::FnScopes::new(fn_def);
        complete_fn(acc, &scopes, ctx.offset);
    }

    if let Some(module) = &ctx.module {
        let module_scope = module.scope(ctx.db)?;
        module_scope
            .entries()
            .filter(|(_name, res)| {
                // Don't expose this item
                match res.import {
                    None => true,
                    Some(import) => {
                        let range = import.range(ctx.db, module.source().file_id());
                        !range.is_subrange(&ctx.leaf.range())
                    }
                }
            })
            .for_each(|(name, _res)| {
                CompletionItem::new(name.to_string())
                    .kind(Reference)
                    .add_to(acc)
            });
    }

    Ok(())
}

fn complete_fn(acc: &mut Completions, scopes: &hir::FnScopes, offset: TextUnit) {
    let mut shadowed = FxHashSet::default();
    scopes
        .scope_chain_for_offset(offset)
        .flat_map(|scope| scopes.entries(scope).iter())
        .filter(|entry| shadowed.insert(entry.name()))
        .for_each(|entry| {
            CompletionItem::new(entry.name().to_string())
                .kind(Reference)
                .add_to(acc)
        });
    if scopes.self_param.is_some() {
        CompletionItem::new("self").kind(Reference).add_to(acc);
    }
}

#[cfg(test)]
mod tests {
    use crate::completion::{CompletionKind, check_completion};

    fn check_reference_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Reference);
    }

    #[test]
    fn completes_bindings_from_let() {
        check_reference_completion(
            r"
            fn quux(x: i32) {
                let y = 92;
                1 + <|>;
                let z = ();
            }
            ",
            "y;x;quux",
        );
    }

    #[test]
    fn completes_bindings_from_if_let() {
        check_reference_completion(
            r"
            fn quux() {
                if let Some(x) = foo() {
                    let y = 92;
                };
                if let Some(a) = bar() {
                    let b = 62;
                    1 + <|>
                }
            }
            ",
            "b;a;quux",
        );
    }

    #[test]
    fn completes_bindings_from_for() {
        check_reference_completion(
            r"
            fn quux() {
                for x in &[1, 2, 3] {
                    <|>
                }
            }
            ",
            "x;quux",
        );
    }

    #[test]
    fn completes_module_items() {
        check_reference_completion(
            r"
            struct Foo;
            enum Baz {}
            fn quux() {
                <|>
            }
            ",
            "quux;Foo;Baz",
        );
    }

    #[test]
    fn completes_module_items_in_nested_modules() {
        check_reference_completion(
            r"
            struct Foo;
            mod m {
                struct Bar;
                fn quux() { <|> }
            }
            ",
            "quux;Bar",
        );
    }

    #[test]
    fn completes_return_type() {
        check_reference_completion(
            r"
            struct Foo;
            fn x() -> <|>
            ",
            "Foo;x",
        )
    }

    #[test]
    fn dont_show_to_completions_for_shadowing() {
        check_reference_completion(
            r"
            fn foo() -> {
                let bar = 92;
                {
                    let bar = 62;
                    <|>
                }
            }
            ",
            "bar;foo",
        )
    }

    #[test]
    fn completes_self_in_methods() {
        check_reference_completion(r"impl S { fn foo(&self) { <|> } }", "self")
    }
}
