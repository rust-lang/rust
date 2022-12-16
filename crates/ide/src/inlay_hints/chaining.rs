use hir::{HirDisplay, Semantics};
use ide_db::{famous_defs::FamousDefs, RootDatabase};
use syntax::{
    ast::{self, AstNode},
    Direction, NodeOrToken, SyntaxKind, T,
};

use crate::{
    inlay_hints::hint_iterator, FileId, InlayHint, InlayHintsConfig, InlayKind, InlayTooltip,
};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<'_, RootDatabase>,
    famous_defs: &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    file_id: FileId,
    expr: &ast::Expr,
) -> Option<()> {
    if !config.chaining_hints {
        return None;
    }

    if matches!(expr, ast::Expr::RecordExpr(_)) {
        return None;
    }

    let descended = sema.descend_node_into_attributes(expr.clone()).pop();
    let desc_expr = descended.as_ref().unwrap_or(expr);

    let mut tokens = expr
        .syntax()
        .siblings_with_tokens(Direction::Next)
        .filter_map(NodeOrToken::into_token)
        .filter(|t| match t.kind() {
            SyntaxKind::WHITESPACE if !t.text().contains('\n') => false,
            SyntaxKind::COMMENT => false,
            _ => true,
        });

    // Chaining can be defined as an expression whose next sibling tokens are newline and dot
    // Ignoring extra whitespace and comments
    let next = tokens.next()?.kind();
    if next == SyntaxKind::WHITESPACE {
        let mut next_next = tokens.next()?.kind();
        while next_next == SyntaxKind::WHITESPACE {
            next_next = tokens.next()?.kind();
        }
        if next_next == T![.] {
            let ty = sema.type_of_expr(desc_expr)?.original;
            if ty.is_unknown() {
                return None;
            }
            if matches!(expr, ast::Expr::PathExpr(_)) {
                if let Some(hir::Adt::Struct(st)) = ty.as_adt() {
                    if st.fields(sema.db).is_empty() {
                        return None;
                    }
                }
            }
            acc.push(InlayHint {
                range: expr.syntax().text_range(),
                kind: InlayKind::ChainingHint,
                label: hint_iterator(sema, &famous_defs, config, &ty)
                    .unwrap_or_else(|| ty.display_truncated(sema.db, config.max_length).to_string())
                    .into(),
                tooltip: Some(InlayTooltip::HoverRanged(file_id, expr.syntax().text_range())),
            });
        }
    }
    Some(())
}
