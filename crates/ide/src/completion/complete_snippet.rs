//! FIXME: write short doc here

use crate::completion::{
    completion_config::SnippetCap, completion_item::Builder, CompletionContext, CompletionItem,
    CompletionItemKind, CompletionKind, Completions,
};

fn snippet(ctx: &CompletionContext, cap: SnippetCap, label: &str, snippet: &str) -> Builder {
    CompletionItem::new(CompletionKind::Snippet, ctx.source_range(), label)
        .insert_snippet(cap, snippet)
        .kind(CompletionItemKind::Snippet)
}

pub(super) fn complete_expr_snippet(acc: &mut Completions, ctx: &CompletionContext) {
    if !(ctx.is_trivial_path && ctx.function_syntax.is_some()) {
        return;
    }
    let cap = match ctx.config.snippet_cap {
        Some(it) => it,
        None => return,
    };

    snippet(ctx, cap, "pd", "eprintln!(\"$0 = {:?}\", $0);").add_to(acc);
    snippet(ctx, cap, "ppd", "eprintln!(\"$0 = {:#?}\", $0);").add_to(acc);
}

pub(super) fn complete_item_snippet(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_new_item {
        return;
    }
    let cap = match ctx.config.snippet_cap {
        Some(it) => it,
        None => return,
    };

    snippet(
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
    )
    .lookup_by("tmod")
    .add_to(acc);

    snippet(
        ctx,
        cap,
        "tfn (Test function)",
        "\
#[test]
fn ${1:feature}() {
    $0
}",
    )
    .lookup_by("tfn")
    .add_to(acc);

    snippet(ctx, cap, "macro_rules", "macro_rules! $1 {\n\t($2) => {\n\t\t$0\n\t};\n}").add_to(acc);
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::completion::{test_utils::completion_list, CompletionKind};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Snippet);
        expect.assert_eq(&actual)
    }

    #[test]
    fn completes_snippets_in_expressions() {
        check(
            r#"fn foo(x: i32) { <|> }"#,
            expect![[r#"
                sn pd
                sn ppd
            "#]],
        );
    }

    #[test]
    fn should_not_complete_snippets_in_path() {
        check(r#"fn foo(x: i32) { ::foo<|> }"#, expect![[""]]);
        check(r#"fn foo(x: i32) { ::<|> }"#, expect![[""]]);
    }

    #[test]
    fn completes_snippets_in_items() {
        check(
            r#"
#[cfg(test)]
mod tests {
    <|>
}
"#,
            expect![[r#"
                sn macro_rules
                sn tfn (Test function)
                sn tmod (Test module)
            "#]],
        )
    }
}
