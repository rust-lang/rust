//! FIXME: write short doc here

use crate::{
    completion::{
        completion_context::CompletionContext,
        completion_item::{Builder, CompletionKind, Completions},
    },
    CompletionItem,
};
use hir::{Ty, TypeCtor};
use ra_syntax::{ast::AstNode, TextRange};
use ra_text_edit::TextEditBuilder;

fn postfix_snippet(ctx: &CompletionContext, label: &str, detail: &str, snippet: &str) -> Builder {
    let edit = {
        let receiver_range =
            ctx.dot_receiver.as_ref().expect("no receiver available").syntax().text_range();
        let delete_range = TextRange::from_to(receiver_range.start(), ctx.source_range().end());
        let mut builder = TextEditBuilder::default();
        builder.replace(delete_range, snippet.to_string());
        builder.finish()
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
        let receiver_text = dot_receiver.syntax().text().to_string();
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
}
