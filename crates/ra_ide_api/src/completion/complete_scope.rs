use rustc_hash::FxHashSet;
use ra_syntax::{AstNode, TextUnit};

use crate::completion::{CompletionItem, CompletionItemKind, Completions, CompletionKind, CompletionContext};

pub(super) fn complete_scope(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_trivial_path {
        return;
    }
    let module = match &ctx.module {
        Some(it) => it,
        None => return,
    };
    if let Some(function) = &ctx.function {
        let scopes = function.scopes(ctx.db);
        complete_fn(acc, &scopes, ctx.offset);
    }

    let module_scope = module.scope(ctx.db);
    module_scope
        .entries()
        .filter(|(_name, res)| {
            // For cases like `use self::foo<|>` don't suggest foo itself.
            match res.import {
                None => true,
                Some(import) => {
                    let source = module.import_source(ctx.db, import);
                    !source.syntax().range().is_subrange(&ctx.leaf.range())
                }
            }
        })
        .for_each(|(name, res)| {
            CompletionItem::new(CompletionKind::Reference, name.to_string())
                .from_resolution(ctx, res)
                .add_to(acc)
        });
}

fn complete_fn(acc: &mut Completions, scopes: &hir::ScopesWithSyntaxMapping, offset: TextUnit) {
    let mut shadowed = FxHashSet::default();
    scopes
        .scope_chain_for_offset(offset)
        .flat_map(|scope| scopes.scopes.entries(scope).iter())
        .filter(|entry| shadowed.insert(entry.name()))
        .for_each(|entry| {
            CompletionItem::new(CompletionKind::Reference, entry.name().to_string())
                .kind(CompletionItemKind::Binding)
                .add_to(acc)
        });
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
            r#"y;x;quux "quux($0)""#,
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
            r#"b;a;quux "quux()$0""#,
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
            r#"x;quux "quux()$0""#,
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
            r#"quux "quux()$0";Foo;Baz"#,
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
            r#"quux "quux()$0";Bar"#,
        );
    }

    #[test]
    fn completes_return_type() {
        check_reference_completion(
            r"
            struct Foo;
            fn x() -> <|>
            ",
            r#"Foo;x "x()$0""#,
        )
    }

    #[test]
    fn dont_show_both_completions_for_shadowing() {
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
            r#"bar;foo "foo()$0""#,
        )
    }

    #[test]
    fn completes_self_in_methods() {
        check_reference_completion(r"impl S { fn foo(&self) { <|> } }", "self")
    }

    #[test]
    fn inserts_parens_for_function_calls() {
        check_reference_completion(
            r"
            fn no_args() {}
            fn main() { no_<|> }
            ",
            r#"no_args "no_args()$0"
               main "main()$0""#,
        );
        check_reference_completion(
            r"
            fn with_args(x: i32, y: String) {}
            fn main() { with_<|> }
            ",
            r#"main "main()$0"
               with_args "with_args($0)""#,
        );
    }
}
