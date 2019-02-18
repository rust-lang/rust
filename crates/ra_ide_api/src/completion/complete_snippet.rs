use crate::completion::{CompletionItem, Completions, CompletionKind, CompletionItemKind, CompletionContext, completion_item::Builder};

fn snippet(ctx: &CompletionContext, label: &str, snippet: &str) -> Builder {
    CompletionItem::new(CompletionKind::Snippet, ctx.source_range(), label)
        .insert_snippet(snippet)
        .kind(CompletionItemKind::Snippet)
}

pub(super) fn complete_expr_snippet(acc: &mut Completions, ctx: &CompletionContext) {
    if !(ctx.is_trivial_path && ctx.function_syntax.is_some()) {
        return;
    }

    snippet(ctx, "pd", "eprintln!(\"$0 = {:?}\", $0);").add_to(acc);
    snippet(ctx, "ppd", "eprintln!(\"$0 = {:#?}\", $0);").add_to(acc);
}

pub(super) fn complete_item_snippet(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_new_item {
        return;
    }
    snippet(
        ctx,
        "Test function",
        "\
#[test]
fn ${1:feature}() {
    $0
}",
    )
    .lookup_by("tfn")
    .add_to(acc);

    snippet(ctx, "pub(crate)", "pub(crate) $0").add_to(acc);
}

#[cfg(test)]
mod tests {
    use crate::completion::CompletionKind;
    use crate::completion::completion_item::check_completion;

    fn check_snippet_completion(name: &str, code: &str) {
        check_completion(name, code, CompletionKind::Snippet);
    }

    #[test]
    fn completes_snippets_in_expressions() {
        check_snippet_completion("snippets_in_expressions", r"fn foo(x: i32) { <|> }");
    }

    #[test]
    fn should_not_complete_snippets_in_path() {
        check_snippet_completion(
            "should_not_complete_snippets_in_path",
            r"fn foo(x: i32) { ::foo<|> }",
        );
        check_snippet_completion(
            "should_not_complete_snippets_in_path2",
            r"fn foo(x: i32) { ::<|> }",
        );
    }

    #[test]
    fn completes_snippets_in_items() {
        check_snippet_completion(
            "snippets_in_items",
            r"
            #[cfg(test)]
            mod tests {
                <|>
            }
            ",
        );
    }
}
