use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir::{intravisit::FnKind, Body, ExprKind, FnDecl, HirId, ImplicitSelfKind};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::Span;

use super::MISSNAMED_GETTERS;

pub fn check_fn(
    cx: &LateContext<'_>,
    kind: FnKind<'_>,
    decl: &FnDecl<'_>,
    body: &Body<'_>,
    span: Span,
    _hir_id: HirId,
) {
    let FnKind::Method(ref ident, sig) = kind else {
            return;
        };

    // Takes only &(mut) self
    if decl.inputs.len() != 1 {
        return;
    }

    let name = ident.name.as_str();

    let name = match sig.decl.implicit_self {
        ImplicitSelfKind::MutRef => {
            let Some(name) = name.strip_suffix("_mut") else {
                    return;
                };
            name
        },
        ImplicitSelfKind::Imm | ImplicitSelfKind::Mut | ImplicitSelfKind::ImmRef => name,
        ImplicitSelfKind::None => return,
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

    let mut expr = block_expr;
    // Accept &<expr>, &mut <expr> and <expr>
    if let ExprKind::AddrOf(_, _, tmp) = expr.kind {
        expr = tmp;
    }
    let (self_data, used_ident) = if_chain! {
        if let ExprKind::Field(self_data, ident) = expr.kind;
        if ident.name.as_str() != name;
        then {
            (self_data, ident)
        } else {
            return;
        }
    };

    let ty = cx.typeck_results().expr_ty(self_data);

    let def = {
        let mut kind = ty.kind();
        loop {
            match kind {
                ty::Adt(def, _) => break def,
                ty::Ref(_, ty, _) => kind = ty.kind(),
                // We don't do tuples because the function name cannot be a number
                _ => return,
            }
        }
    };

    let mut used_field = None;
    let mut correct_field = None;
    for f in def.all_fields() {
        if f.name.as_str() == name {
            correct_field = Some(f);
        }
        if f.name == used_ident.name {
            used_field = Some(f);
        }
    }

    let Some(used_field) = used_field else {
            if cfg!(debug_assertions) {
                panic!("Struct doesn't contain the correct field");
            } else {
                // Don't ICE when possible
                return;
            }
        };
    let Some(correct_field) = correct_field else {
            return;
        };

    if cx.tcx.type_of(used_field.did) == cx.tcx.type_of(correct_field.did) {
        let left_span = block_expr.span.until(used_ident.span);
        let snippet = snippet(cx, left_span, "..");
        let sugg = format!("{snippet}{name}");
        span_lint_and_then(
            cx,
            MISSNAMED_GETTERS,
            span,
            "getter function appears to return the wrong field",
            |diag| {
                diag.span_suggestion(expr_span, "consider using", sugg, Applicability::MaybeIncorrect);
            },
        );
    }
}
