//! FIXME: write short doc here

use ra_syntax::{
    ast::{self, AstNode},
    TextRange, TextUnit,
};
use ra_text_edit::TextEdit;

use crate::{
    completion::{
        completion_context::CompletionContext,
        completion_item::{Builder, CompletionKind, Completions},
    },
    CompletionItem,
};

pub(super) fn complete_postfix(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.options.enable_postfix_completions {
        return;
    }

    let dot_receiver = match &ctx.dot_receiver {
        Some(it) => it,
        None => return,
    };

    let receiver_text =
        get_receiver_text(dot_receiver, ctx.dot_receiver_is_ambiguous_float_literal);

    let receiver_ty = match ctx.sema.type_of_expr(&dot_receiver) {
        Some(it) => it,
        None => return,
    };

    if receiver_ty.is_bool() || receiver_ty.is_unknown() {
        postfix_snippet(
            ctx,
            &dot_receiver,
            "if",
            "if expr {}",
            &format!("if {} {{$0}}", receiver_text),
        )
        .add_to(acc);
        postfix_snippet(
            ctx,
            &dot_receiver,
            "while",
            "while expr {}",
            &format!("while {} {{\n$0\n}}", receiver_text),
        )
        .add_to(acc);
    }

    // !&&&42 is a compiler error, ergo process it before considering the references
    postfix_snippet(ctx, &dot_receiver, "not", "!expr", &format!("!{}", receiver_text)).add_to(acc);

    postfix_snippet(ctx, &dot_receiver, "ref", "&expr", &format!("&{}", receiver_text)).add_to(acc);
    postfix_snippet(ctx, &dot_receiver, "refm", "&mut expr", &format!("&mut {}", receiver_text))
        .add_to(acc);

    // The rest of the postfix completions create an expression that moves an argument,
    // so it's better to consider references now to avoid breaking the compilation
    let dot_receiver = include_references(dot_receiver);
    let receiver_text =
        get_receiver_text(&dot_receiver, ctx.dot_receiver_is_ambiguous_float_literal);

    postfix_snippet(
        ctx,
        &dot_receiver,
        "match",
        "match expr {}",
        &format!("match {} {{\n    ${{1:_}} => {{$0\\}},\n}}", receiver_text),
    )
    .add_to(acc);

    postfix_snippet(
        ctx,
        &dot_receiver,
        "box",
        "Box::new(expr)",
        &format!("Box::new({})", receiver_text),
    )
    .add_to(acc);

    postfix_snippet(ctx, &dot_receiver, "dbg", "dbg!(expr)", &format!("dbg!({})", receiver_text))
        .add_to(acc);
}

fn get_receiver_text(receiver: &ast::Expr, receiver_is_ambiguous_float_literal: bool) -> String {
    if receiver_is_ambiguous_float_literal {
        let text = receiver.syntax().text();
        let without_dot = ..text.len() - TextUnit::of_char('.');
        text.slice(without_dot).to_string()
    } else {
        receiver.to_string()
    }
}

fn include_references(initial_element: &ast::Expr) -> ast::Expr {
    let mut resulting_element = initial_element.clone();
    while let Some(parent_ref_element) =
        resulting_element.syntax().parent().and_then(ast::RefExpr::cast)
    {
        resulting_element = ast::Expr::from(parent_ref_element);
    }
    resulting_element
}

fn postfix_snippet(
    ctx: &CompletionContext,
    receiver: &ast::Expr,
    label: &str,
    detail: &str,
    snippet: &str,
) -> Builder {
    let edit = {
        let receiver_syntax = receiver.syntax();
        let receiver_range = ctx.sema.original_range(receiver_syntax).range;
        let delete_range = TextRange::from_to(receiver_range.start(), ctx.source_range().end());
        TextEdit::replace(delete_range, snippet.to_string())
    };
    CompletionItem::new(CompletionKind::Postfix, ctx.source_range(), label)
        .detail(detail)
        .snippet_edit(edit)
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;

    use crate::completion::{test_utils::do_completion, CompletionItem, CompletionKind};

    fn do_postfix_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Postfix)
    }

    #[test]
    fn postfix_completion_works_for_trivial_path_expression() {
        assert_debug_snapshot!(
            do_postfix_completion(
                r#"
                fn main() {
                    let bar = true;
                    bar.<|>
                }
                "#,
            ),
            @r###"
        [
            CompletionItem {
                label: "box",
                source_range: [89; 89),
                delete: [85; 89),
                insert: "Box::new(bar)",
                detail: "Box::new(expr)",
            },
            CompletionItem {
                label: "dbg",
                source_range: [89; 89),
                delete: [85; 89),
                insert: "dbg!(bar)",
                detail: "dbg!(expr)",
            },
            CompletionItem {
                label: "if",
                source_range: [89; 89),
                delete: [85; 89),
                insert: "if bar {$0}",
                detail: "if expr {}",
            },
            CompletionItem {
                label: "match",
                source_range: [89; 89),
                delete: [85; 89),
                insert: "match bar {\n    ${1:_} => {$0\\},\n}",
                detail: "match expr {}",
            },
            CompletionItem {
                label: "not",
                source_range: [89; 89),
                delete: [85; 89),
                insert: "!bar",
                detail: "!expr",
            },
            CompletionItem {
                label: "ref",
                source_range: [89; 89),
                delete: [85; 89),
                insert: "&bar",
                detail: "&expr",
            },
            CompletionItem {
                label: "refm",
                source_range: [89; 89),
                delete: [85; 89),
                insert: "&mut bar",
                detail: "&mut expr",
            },
            CompletionItem {
                label: "while",
                source_range: [89; 89),
                delete: [85; 89),
                insert: "while bar {\n$0\n}",
                detail: "while expr {}",
            },
        ]
        "###
        );
    }

    #[test]
    fn some_postfix_completions_ignored() {
        assert_debug_snapshot!(
            do_postfix_completion(
                r#"
                fn main() {
                    let bar: u8 = 12;
                    bar.<|>
                }
                "#,
            ),
            @r###"
        [
            CompletionItem {
                label: "box",
                source_range: [91; 91),
                delete: [87; 91),
                insert: "Box::new(bar)",
                detail: "Box::new(expr)",
            },
            CompletionItem {
                label: "dbg",
                source_range: [91; 91),
                delete: [87; 91),
                insert: "dbg!(bar)",
                detail: "dbg!(expr)",
            },
            CompletionItem {
                label: "match",
                source_range: [91; 91),
                delete: [87; 91),
                insert: "match bar {\n    ${1:_} => {$0\\},\n}",
                detail: "match expr {}",
            },
            CompletionItem {
                label: "not",
                source_range: [91; 91),
                delete: [87; 91),
                insert: "!bar",
                detail: "!expr",
            },
            CompletionItem {
                label: "ref",
                source_range: [91; 91),
                delete: [87; 91),
                insert: "&bar",
                detail: "&expr",
            },
            CompletionItem {
                label: "refm",
                source_range: [91; 91),
                delete: [87; 91),
                insert: "&mut bar",
                detail: "&mut expr",
            },
        ]
        "###
        );
    }

    #[test]
    fn postfix_completion_works_for_ambiguous_float_literal() {
        assert_debug_snapshot!(
            do_postfix_completion(
                r#"
                fn main() {
                    42.<|>
                }
                "#,
            ),
            @r###"
        [
            CompletionItem {
                label: "box",
                source_range: [52; 52),
                delete: [49; 52),
                insert: "Box::new(42)",
                detail: "Box::new(expr)",
            },
            CompletionItem {
                label: "dbg",
                source_range: [52; 52),
                delete: [49; 52),
                insert: "dbg!(42)",
                detail: "dbg!(expr)",
            },
            CompletionItem {
                label: "match",
                source_range: [52; 52),
                delete: [49; 52),
                insert: "match 42 {\n    ${1:_} => {$0\\},\n}",
                detail: "match expr {}",
            },
            CompletionItem {
                label: "not",
                source_range: [52; 52),
                delete: [49; 52),
                insert: "!42",
                detail: "!expr",
            },
            CompletionItem {
                label: "ref",
                source_range: [52; 52),
                delete: [49; 52),
                insert: "&42",
                detail: "&expr",
            },
            CompletionItem {
                label: "refm",
                source_range: [52; 52),
                delete: [49; 52),
                insert: "&mut 42",
                detail: "&mut expr",
            },
        ]
        "###
        );
    }

    #[test]
    fn works_in_simple_macro() {
        assert_debug_snapshot!(
            do_postfix_completion(
                r#"
                macro_rules! m { ($e:expr) => { $e } }
                fn main() {
                    let bar: u8 = 12;
                    m!(bar.b<|>)
                }
                "#,
            ),
            @r###"
        [
            CompletionItem {
                label: "box",
                source_range: [149; 150),
                delete: [145; 150),
                insert: "Box::new(bar)",
                detail: "Box::new(expr)",
            },
            CompletionItem {
                label: "dbg",
                source_range: [149; 150),
                delete: [145; 150),
                insert: "dbg!(bar)",
                detail: "dbg!(expr)",
            },
            CompletionItem {
                label: "match",
                source_range: [149; 150),
                delete: [145; 150),
                insert: "match bar {\n    ${1:_} => {$0\\},\n}",
                detail: "match expr {}",
            },
            CompletionItem {
                label: "not",
                source_range: [149; 150),
                delete: [145; 150),
                insert: "!bar",
                detail: "!expr",
            },
            CompletionItem {
                label: "ref",
                source_range: [149; 150),
                delete: [145; 150),
                insert: "&bar",
                detail: "&expr",
            },
            CompletionItem {
                label: "refm",
                source_range: [149; 150),
                delete: [145; 150),
                insert: "&mut bar",
                detail: "&mut expr",
            },
        ]
        "###
        );
    }

    #[test]
    fn postfix_completion_for_references() {
        assert_debug_snapshot!(
            do_postfix_completion(
                r#"
                fn main() {
                    &&&&42.<|>
                }
                "#,
            ),
            @r###"
        [
            CompletionItem {
                label: "box",
                source_range: [56; 56),
                delete: [49; 56),
                insert: "Box::new(&&&&42)",
                detail: "Box::new(expr)",
            },
            CompletionItem {
                label: "dbg",
                source_range: [56; 56),
                delete: [49; 56),
                insert: "dbg!(&&&&42)",
                detail: "dbg!(expr)",
            },
            CompletionItem {
                label: "match",
                source_range: [56; 56),
                delete: [49; 56),
                insert: "match &&&&42 {\n    ${1:_} => {$0\\},\n}",
                detail: "match expr {}",
            },
            CompletionItem {
                label: "not",
                source_range: [56; 56),
                delete: [53; 56),
                insert: "!42",
                detail: "!expr",
            },
            CompletionItem {
                label: "ref",
                source_range: [56; 56),
                delete: [53; 56),
                insert: "&42",
                detail: "&expr",
            },
            CompletionItem {
                label: "refm",
                source_range: [56; 56),
                delete: [53; 56),
                insert: "&mut 42",
                detail: "&mut expr",
            },
        ]
        "###
        );
    }
}
