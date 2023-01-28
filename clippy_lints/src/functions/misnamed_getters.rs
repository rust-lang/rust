use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir::{intravisit::FnKind, Body, ExprKind, FnDecl, ImplicitSelfKind, Unsafety};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::Span;

use std::iter;

use super::MISNAMED_GETTERS;

pub fn check_fn(
    cx: &LateContext<'_>,
    kind: FnKind<'_>,
    decl: &FnDecl<'_>,
    body: &Body<'_>,
    span: Span,
) {
    let FnKind::Method(ref ident, sig) = kind else {
            return;
        };

    // Takes only &(mut) self
    if decl.inputs.len() != 1 {
        return;
    }

    let name = ident.name.as_str();

    let name = match decl.implicit_self {
        ImplicitSelfKind::MutRef => {
            let Some(name) = name.strip_suffix("_mut") else {
                    return;
                };
            name
        },
        ImplicitSelfKind::Imm | ImplicitSelfKind::Mut | ImplicitSelfKind::ImmRef => name,
        ImplicitSelfKind::None => return,
    };

    let name = if sig.header.unsafety == Unsafety::Unsafe {
        name.strip_suffix("_unchecked").unwrap_or(name)
    } else {
        name
    };

    // Body must be &(mut) <self_data>.name
    // self_data is not neccessarilly self, to also lint sub-getters, etc…

    let block_expr = if_chain! {
        if let ExprKind::Block(block,_) = body.value.kind;
        if block.stmts.is_empty();
        if let Some(block_expr) = block.expr;
        then {
            block_expr
        } else {
            return;
        }
    };
    let expr_span = block_expr.span;

    // Accept &<expr>, &mut <expr> and <expr>
    let expr = if let ExprKind::AddrOf(_, _, tmp) = block_expr.kind {
        tmp
    } else {
        block_expr
    };
    let (self_data, used_ident) = if_chain! {
        if let ExprKind::Field(self_data, ident) = expr.kind;
        if ident.name.as_str() != name;
        then {
            (self_data, ident)
        } else {
            return;
        }
    };

    let mut used_field = None;
    let mut correct_field = None;
    let typeck_results = cx.typeck_results();
    for adjusted_type in iter::once(typeck_results.expr_ty(self_data))
        .chain(typeck_results.expr_adjustments(self_data).iter().map(|adj| adj.target))
    {
        let ty::Adt(def,_) = adjusted_type.kind() else {
            continue;
        };

        for f in def.all_fields() {
            if f.name.as_str() == name {
                correct_field = Some(f);
            }
            if f.name == used_ident.name {
                used_field = Some(f);
            }
        }
    }

    let Some(used_field) = used_field else {
        // Can happen if the field access is a tuple. We don't lint those because the getter name could not start with a number.
        return;
    };

    let Some(correct_field) = correct_field else {
        // There is no field corresponding to the getter name.
        // FIXME: This can be a false positive if the correct field is reachable trought deeper autodereferences than used_field is
        return;
    };

    if cx.tcx.type_of(used_field.did) == cx.tcx.type_of(correct_field.did) {
        let left_span = block_expr.span.until(used_ident.span);
        let snippet = snippet(cx, left_span, "..");
        let sugg = format!("{snippet}{name}");
        span_lint_and_then(
            cx,
            MISNAMED_GETTERS,
            span,
            "getter function appears to return the wrong field",
            |diag| {
                diag.span_suggestion(expr_span, "consider using", sugg, Applicability::MaybeIncorrect);
            },
        );
    }
}
