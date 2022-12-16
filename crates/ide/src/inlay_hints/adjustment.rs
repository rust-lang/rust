use hir::{Adjust, AutoBorrow, Mutability, OverloadedDeref, PointerCast, Safety, Semantics};
use ide_db::RootDatabase;

use syntax::ast::{self, AstNode};

use crate::{AdjustmentHints, InlayHint, InlayHintsConfig, InlayKind};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<'_, RootDatabase>,
    config: &InlayHintsConfig,
    expr: &ast::Expr,
) -> Option<()> {
    if config.adjustment_hints == AdjustmentHints::Never {
        return None;
    }

    // These inherit from the inner expression which would result in duplicate hints
    if let ast::Expr::ParenExpr(_)
    | ast::Expr::IfExpr(_)
    | ast::Expr::BlockExpr(_)
    | ast::Expr::MatchExpr(_) = expr
    {
        return None;
    }

    let parent = expr.syntax().parent().and_then(ast::Expr::cast);
    let descended = sema.descend_node_into_attributes(expr.clone()).pop();
    let desc_expr = descended.as_ref().unwrap_or(expr);
    let adjustments = sema.expr_adjustments(desc_expr).filter(|it| !it.is_empty())?;
    let needs_parens = match parent {
        Some(parent) => {
            match parent {
                ast::Expr::AwaitExpr(_)
                | ast::Expr::CallExpr(_)
                | ast::Expr::CastExpr(_)
                | ast::Expr::FieldExpr(_)
                | ast::Expr::MethodCallExpr(_)
                | ast::Expr::TryExpr(_) => true,
                // FIXME: shorthands need special casing, though not sure if adjustments are even valid there
                ast::Expr::RecordExpr(_) => false,
                ast::Expr::IndexExpr(index) => index.base().as_ref() == Some(expr),
                _ => false,
            }
        }
        None => false,
    };
    if needs_parens {
        acc.push(InlayHint {
            range: expr.syntax().text_range(),
            kind: InlayKind::OpeningParenthesis,
            label: "(".into(),
            tooltip: None,
        });
    }
    for adjustment in adjustments.into_iter().rev() {
        // FIXME: Add some nicer tooltips to each of these
        let text = match adjustment {
            Adjust::NeverToAny if config.adjustment_hints == AdjustmentHints::Always => {
                "<never-to-any>"
            }
            Adjust::Deref(None) => "*",
            Adjust::Deref(Some(OverloadedDeref(Mutability::Mut))) => "*",
            Adjust::Deref(Some(OverloadedDeref(Mutability::Shared))) => "*",
            Adjust::Borrow(AutoBorrow::Ref(Mutability::Shared)) => "&",
            Adjust::Borrow(AutoBorrow::Ref(Mutability::Mut)) => "&mut ",
            Adjust::Borrow(AutoBorrow::RawPtr(Mutability::Shared)) => "&raw const ",
            Adjust::Borrow(AutoBorrow::RawPtr(Mutability::Mut)) => "&raw mut ",
            // some of these could be represented via `as` casts, but that's not too nice and
            // handling everything as a prefix expr makes the `(` and `)` insertion easier
            Adjust::Pointer(cast) if config.adjustment_hints == AdjustmentHints::Always => {
                match cast {
                    PointerCast::ReifyFnPointer => "<fn-item-to-fn-pointer>",
                    PointerCast::UnsafeFnPointer => "<safe-fn-pointer-to-unsafe-fn-pointer>",
                    PointerCast::ClosureFnPointer(Safety::Unsafe) => {
                        "<closure-to-unsafe-fn-pointer>"
                    }
                    PointerCast::ClosureFnPointer(Safety::Safe) => "<closure-to-fn-pointer>",
                    PointerCast::MutToConstPointer => "<mut-ptr-to-const-ptr>",
                    PointerCast::ArrayToPointer => "<array-ptr-to-element-ptr>",
                    PointerCast::Unsize => "<unsize>",
                }
            }
            _ => continue,
        };
        acc.push(InlayHint {
            range: expr.syntax().text_range(),
            kind: InlayKind::AdjustmentHint,
            label: text.into(),
            tooltip: None,
        });
    }
    if needs_parens {
        acc.push(InlayHint {
            range: expr.syntax().text_range(),
            kind: InlayKind::ClosingParenthesis,
            label: ")".into(),
            tooltip: None,
        });
    }
    Some(())
}
