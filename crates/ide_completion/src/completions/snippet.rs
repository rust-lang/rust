//! This file provides snippet completions, like `pd` => `eprintln!(...)`.

use ide_db::helpers::SnippetCap;
use syntax::T;

use crate::{
    context::PathCompletionContext, item::Builder, CompletionContext, CompletionItem,
    CompletionItemKind, CompletionKind, Completions,
};

fn snippet(ctx: &CompletionContext, cap: SnippetCap, label: &str, snippet: &str) -> Builder {
    let mut item = CompletionItem::new(CompletionKind::Snippet, ctx.source_range(), label);
    item.insert_snippet(cap, snippet).kind(CompletionItemKind::Snippet);
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

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{tests::filtered_completion_list, CompletionKind};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = filtered_completion_list(ra_fixture, CompletionKind::Snippet);
        expect.assert_eq(&actual)
    }

    #[test]
    fn completes_snippets_in_expressions() {
        check(
            r#"fn foo(x: i32) { $0 }"#,
            expect![[r#"
                sn pd
                sn ppd
            "#]],
        );
    }

    #[test]
    fn should_not_complete_snippets_in_path() {
        check(r#"fn foo(x: i32) { ::foo$0 }"#, expect![[""]]);
        check(r#"fn foo(x: i32) { ::$0 }"#, expect![[""]]);
    }
}
