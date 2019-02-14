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
    let receiver_range = ctx.dot_receiver.expect("no receiver available").syntax().range();
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
        postfix_snippet(ctx, "not", &format!("!{}", receiver_text)).add_to(acc);
        postfix_snippet(ctx, "ref", &format!("&{}", receiver_text)).add_to(acc);
        postfix_snippet(ctx, "if", &format!("if {} {{$0}}", receiver_text)).add_to(acc);
        postfix_snippet(
            ctx,
            "match",
            &format!("match {} {{\n${{1:_}} => {{$0\\}},\n}}", receiver_text),
        )
        .add_to(acc);
        postfix_snippet(ctx, "while", &format!("while {} {{\n$0\n}}", receiver_text)).add_to(acc);
        postfix_snippet(ctx, "dbg", &format!("dbg!({})", receiver_text)).add_to(acc);
    }
}

#[cfg(test)]
mod tests {
    use crate::completion::completion_item::CompletionKind;
    use crate::completion::completion_item::check_completion;

    fn check_snippet_completion(test_name: &str, code: &str) {
        check_completion(test_name, code, CompletionKind::Postfix);
    }

    #[test]
    fn postfix_completion_works_for_trivial_path_expression() {
        check_snippet_completion(
            "postfix_completion_works_for_trivial_path_expression",
            r#"
            fn main() {
                let bar = "a";
                bar.<|>
            }
            "#,
        );
    }
}
