//! FIXME: write short doc here

use crate::{
    completion::{
        completion_context::CompletionContext,
        completion_item::{Builder, CompletionKind, Completions},
    },
    CompletionItem,
};
use hir::{Ty, TypeCtor};
use ra_syntax::{ast::AstNode, TextRange, TextUnit};
use ra_text_edit::TextEdit;

fn postfix_snippet(ctx: &CompletionContext, label: &str, detail: &str, snippet: &str) -> Builder {
    let edit = {
        let receiver_range =
            ctx.dot_receiver.as_ref().expect("no receiver available").syntax().text_range();
        let delete_range = TextRange::from_to(receiver_range.start(), ctx.source_range().end());
        TextEdit::replace(delete_range, snippet.to_string())
    };
    CompletionItem::new(CompletionKind::Postfix, ctx.source_range(), label)
        .detail(detail)
        .snippet_edit(edit)
}

fn is_bool_or_unknown(ty: Option<Ty>) -> bool {
    if let Some(ty) = ty {
        match ty {
            Ty::Apply(at) => match at.ctor {
                TypeCtor::Bool => true,
                _ => false,
            },
            Ty::Unknown => true,
            _ => false,
        }
    } else {
        true
    }
}

pub(super) fn complete_postfix(acc: &mut Completions, ctx: &CompletionContext) {
    if let Some(dot_receiver) = &ctx.dot_receiver {
        let receiver_text = if ctx.dot_receiver_is_ambiguous_float_literal {
            let text = dot_receiver.syntax().text();
            let without_dot = ..text.len() - TextUnit::of_char('.');
            text.slice(without_dot).to_string()
        } else {
            dot_receiver.syntax().text().to_string()
        };
        let receiver_ty = ctx.analyzer.type_of(ctx.db, &dot_receiver);
        if is_bool_or_unknown(receiver_ty) {
            postfix_snippet(ctx, "if", "if expr {}", &format!("if {} {{$0}}", receiver_text))
                .add_to(acc);
            postfix_snippet(
                ctx,
                "while",
                "while expr {}",
                &format!("while {} {{\n$0\n}}", receiver_text),
            )
            .add_to(acc);
        }
        postfix_snippet(ctx, "not", "!expr", &format!("!{}", receiver_text)).add_to(acc);
        postfix_snippet(ctx, "ref", "&expr", &format!("&{}", receiver_text)).add_to(acc);
        postfix_snippet(ctx, "refm", "&mut expr", &format!("&mut {}", receiver_text)).add_to(acc);
        postfix_snippet(
            ctx,
            "match",
            "match expr {}",
            &format!("match {} {{\n    ${{1:_}} => {{$0\\}},\n}}", receiver_text),
        )
        .add_to(acc);
        postfix_snippet(ctx, "dbg", "dbg!(expr)", &format!("dbg!({})", receiver_text)).add_to(acc);
        postfix_snippet(ctx, "box", "Box::new(expr)", &format!("Box::new({})", receiver_text))
            .add_to(acc);
    }
}

#[cfg(test)]
mod tests {
    use crate::completion::{do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot;

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
            @r###"[
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
]"###
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
            @r###"[
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
]"###
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
            @r###"[
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
]"###
        );
    }
}
