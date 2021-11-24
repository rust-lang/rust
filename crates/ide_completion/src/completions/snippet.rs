//! This file provides snippet completions, like `pd` => `eprintln!(...)`.

use hir::Documentation;
use ide_db::helpers::{insert_use::ImportScope, SnippetCap};
use syntax::T;

use crate::{
    context::PathCompletionContext, item::Builder, CompletionContext, CompletionItem,
    CompletionItemKind, Completions, SnippetScope,
};

fn snippet(ctx: &CompletionContext, cap: SnippetCap, label: &str, snippet: &str) -> Builder {
    let mut item = CompletionItem::new(CompletionItemKind::Snippet, ctx.source_range(), label);
    item.insert_snippet(cap, snippet);
    item
}

pub(crate) fn complete_expr_snippet(acc: &mut Completions, ctx: &CompletionContext) {
    if ctx.function_def.is_none() {
        return;
    }

    let can_be_stmt = match ctx.path_context {
        Some(PathCompletionContext { is_trivial_path: true, can_be_stmt, .. }) => can_be_stmt,
        _ => return,
    };

    let cap = match ctx.config.snippet_cap {
        Some(it) => it,
        None => return,
    };

    if !ctx.config.snippets.is_empty() {
        add_custom_completions(acc, ctx, cap, SnippetScope::Expr);
    }

    if can_be_stmt {
        snippet(ctx, cap, "pd", "eprintln!(\"$0 = {:?}\", $0);").add_to(acc);
        snippet(ctx, cap, "ppd", "eprintln!(\"$0 = {:#?}\", $0);").add_to(acc);
    }
}

pub(crate) fn complete_item_snippet(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.expects_item()
        || ctx.previous_token_is(T![unsafe])
        || ctx.path_qual().is_some()
        || ctx.has_impl_or_trait_prev_sibling()
    {
        return;
    }
    if ctx.has_visibility_prev_sibling() {
        return; // technically we could do some of these snippet completions if we were to put the
                // attributes before the vis node.
    }
    let cap = match ctx.config.snippet_cap {
        Some(it) => it,
        None => return,
    };

    if !ctx.config.snippets.is_empty() {
        add_custom_completions(acc, ctx, cap, SnippetScope::Item);
    }

    let mut item = snippet(
        ctx,
        cap,
        "tmod (Test module)",
        "\
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ${1:test_name}() {
        $0
    }
}",
    );
    item.lookup_by("tmod");
    item.add_to(acc);

    let mut item = snippet(
        ctx,
        cap,
        "tfn (Test function)",
        "\
#[test]
fn ${1:feature}() {
    $0
}",
    );
    item.lookup_by("tfn");
    item.add_to(acc);

    let item = snippet(ctx, cap, "macro_rules", "macro_rules! $1 {\n\t($2) => {\n\t\t$0\n\t};\n}");
    item.add_to(acc);
}

fn add_custom_completions(
    acc: &mut Completions,
    ctx: &CompletionContext,
    cap: SnippetCap,
    scope: SnippetScope,
) -> Option<()> {
    let import_scope = ImportScope::find_insert_use_container(&ctx.token.parent()?, &ctx.sema)?;
    ctx.config.prefix_snippets().filter(|(_, snip)| snip.scope == scope).for_each(
        |(trigger, snip)| {
            let imports = match snip.imports(ctx, &import_scope) {
                Some(imports) => imports,
                None => return,
            };
            let body = snip.snippet();
            let mut builder = snippet(ctx, cap, &trigger, &body);
            builder.documentation(Documentation::new(format!("```rust\n{}\n```", body)));
            for import in imports.into_iter() {
                builder.add_import(import);
            }
            builder.set_detail(snip.description.clone());
            builder.add_to(acc);
        },
    );
    None
}

#[cfg(test)]
mod tests {
    use crate::{
        tests::{check_edit_with_config, TEST_CONFIG},
        CompletionConfig, Snippet,
    };

    #[test]
    fn custom_snippet_completion() {
        check_edit_with_config(
            CompletionConfig {
                snippets: vec![Snippet::new(
                    &["break".into()],
                    &[],
                    &["ControlFlow::Break(())".into()],
                    "",
                    &["core::ops::ControlFlow".into()],
                    crate::SnippetScope::Expr,
                )
                .unwrap()],
                ..TEST_CONFIG
            },
            "break",
            r#"
//- minicore: try
fn main() { $0 }
"#,
            r#"
use core::ops::ControlFlow;

fn main() { ControlFlow::Break(()) }
"#,
        );
    }
}
