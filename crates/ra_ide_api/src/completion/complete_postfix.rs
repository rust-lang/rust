use crate::{
    completion::{
        completion_item::{
            Completions,
            Builder,
            CompletionKind,
        },
        completion_context::CompletionContext,
    },
    CompletionItem
};
use ra_syntax::{
    ast::AstNode,
    TextRange
};
use ra_text_edit::TextEditBuilder;

fn postfix_snippet(ctx: &CompletionContext, label: &str, snippet: &str) -> Builder {
    let replace_range = ctx.source_range();
    let receiver_range = ctx
        .dot_receiver
        .expect("no receiver available")
        .syntax()
        .range();
    let delete_range = TextRange::from_to(receiver_range.start(), replace_range.start());
    let mut builder = TextEditBuilder::default();
    builder.delete(delete_range);
    CompletionItem::new(CompletionKind::Postfix, replace_range, label)
        .snippet(snippet)
        .text_edit(builder.finish())
}

pub(super) fn complete_postfix(acc: &mut Completions, ctx: &CompletionContext) {
    if let Some(dot_receiver) = ctx.dot_receiver {
        let receiver_text = dot_receiver.syntax().text().to_string();
        postfix_snippet(ctx, "not", "!not").add_to(acc);
        postfix_snippet(ctx, "if", &format!("if {} {{$0}}", receiver_text)).add_to(acc);
        postfix_snippet(
            ctx,
            "match",
            &format!("match {} {{\n${{1:_}} => {{$0\\}},\n}}", receiver_text),
        )
        .add_to(acc);
        postfix_snippet(ctx, "while", &format!("while {} {{\n$0\n}}", receiver_text)).add_to(acc);
    }
}

#[cfg(test)]
mod tests {
    use crate::completion::completion_item::CompletionKind;
    use crate::completion::completion_item::check_completion;

    fn check_snippet_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Postfix);
    }

    #[test]
    fn test_filter_postfix_completion1() {
        check_snippet_completion(
            "filter_postfix_completion1",
            r#"
            fn main() {
                let bar = "a";
                bar.<|>
            }
            "#,
        );
    }

    #[test]
    fn test_filter_postfix_completion2() {
        check_snippet_completion(
            "filter_postfix_completion2",
            r#"
            fn main() {
                let bar = "a";
                bar.i<|>
            }
            "#,
        );
    }

    #[test]
    fn test_filter_postfix_completion3() {
        check_snippet_completion(
            "filter_postfix_completion3",
            r#"
            fn main() {
                let bar = "a";
                bar.if<|>
            }
            "#,
        );
    }
}
