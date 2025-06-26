//! This file provides snippet completions, like `pd` => `eprintln!(...)`.

use ide_db::{SnippetCap, documentation::Documentation, imports::insert_use::ImportScope};

use crate::{
    CompletionContext, CompletionItem, CompletionItemKind, Completions, SnippetScope,
    context::{ItemListKind, PathCompletionCtx, PathExprCtx, Qualified},
    item::Builder,
};

pub(crate) fn complete_expr_snippet(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx: &PathCompletionCtx<'_>,
    &PathExprCtx { in_block_expr, .. }: &PathExprCtx<'_>,
) {
    if !matches!(path_ctx.qualified, Qualified::No) {
        return;
    }
    if !ctx.qualifier_ctx.none() {
        return;
    }

    let cap = match ctx.config.snippet_cap {
        Some(it) => it,
        None => return,
    };

    if !ctx.config.snippets.is_empty() {
        add_custom_completions(acc, ctx, cap, SnippetScope::Expr);
    }

    if in_block_expr {
        snippet(ctx, cap, "pd", "eprintln!(\"$0 = {:?}\", $0);").add_to(acc, ctx.db);
        snippet(ctx, cap, "ppd", "eprintln!(\"$0 = {:#?}\", $0);").add_to(acc, ctx.db);
        let item = snippet(
            ctx,
            cap,
            "macro_rules",
            "\
macro_rules! $1 {
    ($2) => {
        $0
    };
}",
        );
        item.add_to(acc, ctx.db);
    }
}

pub(crate) fn complete_item_snippet(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx: &PathCompletionCtx<'_>,
    kind: &ItemListKind,
) {
    if !matches!(path_ctx.qualified, Qualified::No) {
        return;
    }
    if !ctx.qualifier_ctx.none() {
        return;
    }
    let cap = match ctx.config.snippet_cap {
        Some(it) => it,
        None => return,
    };

    if !ctx.config.snippets.is_empty() {
        add_custom_completions(acc, ctx, cap, SnippetScope::Item);
    }

    // Test-related snippets shouldn't be shown in blocks.
    if let ItemListKind::SourceFile | ItemListKind::Module = kind {
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
        item.add_to(acc, ctx.db);

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
        item.add_to(acc, ctx.db);

        let item = snippet(
            ctx,
            cap,
            "macro_rules",
            "\
macro_rules! $1 {
    ($2) => {
        $0
    };
}",
        );
        item.add_to(acc, ctx.db);
    }
}

fn snippet(ctx: &CompletionContext<'_>, cap: SnippetCap, label: &str, snippet: &str) -> Builder {
    let mut item =
        CompletionItem::new(CompletionItemKind::Snippet, ctx.source_range(), label, ctx.edition);
    item.insert_snippet(cap, snippet);
    item
}

fn add_custom_completions(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    cap: SnippetCap,
    scope: SnippetScope,
) -> Option<()> {
    ImportScope::find_insert_use_container(&ctx.token.parent()?, &ctx.sema)?;
    ctx.config.prefix_snippets().filter(|(_, snip)| snip.scope == scope).for_each(
        |(trigger, snip)| {
            let imports = match snip.imports(ctx) {
                Some(imports) => imports,
                None => return,
            };
            let body = snip.snippet();
            let mut builder = snippet(ctx, cap, trigger, &body);
            builder.documentation(Documentation::new(format!("```rust\n{body}\n```")));
            for import in imports.into_iter() {
                builder.add_import(import);
            }
            builder.set_detail(snip.description.clone());
            builder.add_to(acc, ctx.db);
        },
    );
    None
}

#[cfg(test)]
mod tests {
    use crate::{
        CompletionConfig, Snippet,
        tests::{TEST_CONFIG, check_edit_with_config},
    };

    #[test]
    fn custom_snippet_completion() {
        check_edit_with_config(
            CompletionConfig {
                snippets: vec![
                    Snippet::new(
                        &["break".into()],
                        &[],
                        &["ControlFlow::Break(())".into()],
                        "",
                        &["core::ops::ControlFlow".into()],
                        crate::SnippetScope::Expr,
                    )
                    .unwrap(),
                ],
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
