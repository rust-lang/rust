use crate::completion::{CompletionContext, CompletionItem, Completions, CompletionKind, CompletionItemKind};

pub(super) fn complete_use_tree_keyword(acc: &mut Completions, ctx: &CompletionContext) {
    // complete keyword "crate" in use stmt
    match (ctx.use_item_syntax.as_ref(), ctx.path_prefix.as_ref()) {
        (Some(_), None) => {
            CompletionItem::new(CompletionKind::Keyword, "crate")
                .kind(CompletionItemKind::Keyword)
                .lookup_by("crate")
                .snippet("crate::")
                .add_to(acc);
            CompletionItem::new(CompletionKind::Keyword, "self")
                .kind(CompletionItemKind::Keyword)
                .lookup_by("self")
                .add_to(acc);
            CompletionItem::new(CompletionKind::Keyword, "super")
                .kind(CompletionItemKind::Keyword)
                .lookup_by("super")
                .add_to(acc);
        }
        (Some(_), Some(_)) => {
            CompletionItem::new(CompletionKind::Keyword, "self")
                .kind(CompletionItemKind::Keyword)
                .lookup_by("self")
                .add_to(acc);
            CompletionItem::new(CompletionKind::Keyword, "super")
                .kind(CompletionItemKind::Keyword)
                .lookup_by("super")
                .add_to(acc);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use crate::completion::{CompletionKind, check_completion};
    fn check_keyword_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Keyword);
    }

    #[test]
    fn completes_keywords_in_use_stmt() {
        check_keyword_completion(
            r"
            use <|>
            ",
            r#"
            crate "crate" "crate::"
            self "self"
            super "super"
            "#,
        );

        check_keyword_completion(
            r"
            use a::<|>
            ",
            r#"
            self "self"
            super "super"
            "#,
        );

        check_keyword_completion(
            r"
            use a::{b, <|>}
            ",
            r#"
            self "self"
            super "super"
            "#,
        );
    }
}
