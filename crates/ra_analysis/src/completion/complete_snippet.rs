use crate::completion::{CompletionItem, Completions, CompletionKind::*, CompletionContext};

pub(super) fn complete_expr_snippet(acc: &mut Completions, ctx: &CompletionContext) {
    if !(ctx.is_trivial_path && ctx.enclosing_fn.is_some()) {
        return;
    }
    CompletionItem::new(Snippet, "pd")
        .snippet("eprintln!(\"$0 = {:?}\", $0);")
        .add_to(acc);
    CompletionItem::new(Snippet, "ppd")
        .snippet("eprintln!(\"$0 = {:#?}\", $0);")
        .add_to(acc);
}

pub(super) fn complete_item_snippet(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_new_item {
        return;
    }
    CompletionItem::new(Snippet, "Test function")
        .lookup_by("tfn")
        .snippet(
            "\
#[test]
fn ${1:feature}() {
    $0
}",
        )
        .add_to(acc);
    CompletionItem::new(Snippet, "pub(crate)")
        .snippet("pub(crate) $0")
        .add_to(acc);
}

#[cfg(test)]
mod tests {
    use crate::completion::{CompletionKind, check_completion};
    fn check_snippet_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Snippet);
    }

    #[test]
    fn completes_snippets_in_expressions() {
        check_snippet_completion(
            r"fn foo(x: i32) { <|> }",
            r##"
            pd "eprintln!(\"$0 = {:?}\", $0);"
            ppd "eprintln!(\"$0 = {:#?}\", $0);"
            "##,
        );
    }

    #[test]
    fn completes_snippets_in_items() {
        // check_snippet_completion(r"
        //     <|>
        //     ",
        //     r##"[CompletionItem { label: "Test function", lookup: None, snippet: Some("#[test]\nfn test_${1:feature}() {\n$0\n}"##,
        // );
        check_snippet_completion(
            r"
            #[cfg(test)]
            mod tests {
                <|>
            }
            ",
            r##"
            tfn "Test function" "#[test]\nfn ${1:feature}() {\n    $0\n}"
            pub(crate) "pub(crate) $0"
            "##,
        );
    }
}
