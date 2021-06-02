//! This file provides snippet completions, like `pd` => `eprintln!(...)`.

use ide_db::helpers::SnippetCap;

use crate::{
    item::Builder, CompletionContext, CompletionItem, CompletionItemKind, CompletionKind,
    Completions,
};

fn snippet(ctx: &CompletionContext, cap: SnippetCap, label: &str, snippet: &str) -> Builder {
    let mut item = CompletionItem::new(CompletionKind::Snippet, ctx.source_range(), label);
    item.insert_snippet(cap, snippet).kind(CompletionItemKind::Snippet);
    item
}

pub(crate) fn complete_expr_snippet(acc: &mut Completions, ctx: &CompletionContext) {
    if !(ctx.is_trivial_path && ctx.function_def.is_some()) {
        return;
    }
    let cap = match ctx.config.snippet_cap {
        Some(it) => it,
        None => return,
    };

    if ctx.can_be_stmt {
        snippet(ctx, cap, "pd", "eprintln!(\"$0 = {:?}\", $0);").add_to(acc);
        snippet(ctx, cap, "ppd", "eprintln!(\"$0 = {:#?}\", $0);").add_to(acc);
    }
}

pub(crate) fn complete_item_snippet(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.expects_item() {
        return;
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

    use crate::{test_utils::completion_list, CompletionKind};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Snippet);
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

    #[test]
    fn completes_snippets_in_items() {
        check(
            r#"
#[cfg(test)]
mod tests {
    $0
}
"#,
            expect![[r#"
                sn tmod (Test module)
                sn tfn (Test function)
                sn macro_rules
            "#]],
        )
    }
}
