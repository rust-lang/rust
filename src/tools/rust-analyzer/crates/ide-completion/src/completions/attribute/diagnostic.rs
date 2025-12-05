//! Completion for diagnostic attributes.

use ide_db::SymbolKind;
use syntax::ast;

use crate::{CompletionItem, Completions, context::CompletionContext};

use super::AttrCompletion;

pub(super) fn complete_on_unimplemented(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    input: ast::TokenTree,
) {
    if let Some(existing_keys) = super::parse_comma_sep_expr(input) {
        for attr in ATTRIBUTE_ARGS {
            let already_annotated = existing_keys
                .iter()
                .filter_map(|expr| match expr {
                    ast::Expr::PathExpr(path) => path.path()?.as_single_name_ref(),
                    ast::Expr::BinExpr(bin)
                        if bin.op_kind() == Some(ast::BinaryOp::Assignment { op: None }) =>
                    {
                        match bin.lhs()? {
                            ast::Expr::PathExpr(path) => path.path()?.as_single_name_ref(),
                            _ => None,
                        }
                    }
                    _ => None,
                })
                .any(|it| {
                    let text = it.text();
                    attr.key() == text && text != "note"
                });
            if already_annotated {
                continue;
            }

            let mut item = CompletionItem::new(
                SymbolKind::BuiltinAttr,
                ctx.source_range(),
                attr.label,
                ctx.edition,
            );
            if let Some(lookup) = attr.lookup {
                item.lookup_by(lookup);
            }
            if let Some((snippet, cap)) = attr.snippet.zip(ctx.config.snippet_cap) {
                item.insert_snippet(cap, snippet);
            }
            item.add_to(acc, ctx.db);
        }
    }
}

const ATTRIBUTE_ARGS: &[AttrCompletion] = &[
    super::attr(r#"label = "…""#, Some("label"), Some(r#"label = "${0:label}""#)),
    super::attr(r#"message = "…""#, Some("message"), Some(r#"message = "${0:message}""#)),
    super::attr(r#"note = "…""#, Some("note"), Some(r#"note = "${0:note}""#)),
];
