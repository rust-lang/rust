use clippy_utils::diagnostics::span_lint_and_sugg;
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
    _span: Span,
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
        ImplicitSelfKind::ImmRef => name,
        ImplicitSelfKind::MutRef => {
            let Some(name) = name.strip_suffix("_mut") else {
                    return;
                };
            name
        },
        _ => return,
    };

    // Body must be &(mut) <self_data>.name
    // self_data is not neccessarilly self
    let (self_data, used_ident, span) = if_chain! {
        if let ExprKind::Block(block,_) = body.value.kind;
        if block.stmts.is_empty();
        if let Some(block_expr) = block.expr;
        // replace with while for as many addrof needed
        if let ExprKind::AddrOf(_,_, expr) = block_expr.kind;
        if let ExprKind::Field(self_data, ident) = expr.kind;
        if ident.name.as_str() != name;
        then {
            (self_data,ident,block_expr.span)
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

    let variants = def.variants();

    // We're accessing a field, so it should be an union or a struct and have one and only one variant
    if variants.len() != 1 {
        if cfg!(debug_assertions) {
            panic!("Struct or union expected to have only one variant");
        } else {
            // Don't ICE when possible
            return;
        }
    }

    let first = variants.last().unwrap();
    let fields = &variants[first];

    let mut used_field = None;
    let mut correct_field = None;
    for f in &fields.fields {
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
        let snippet = snippet(cx, span, "..");
        let sugg = format!("{}{name}", snippet.strip_suffix(used_field.name.as_str()).unwrap());
        span_lint_and_sugg(
            cx,
            MISSNAMED_GETTERS,
            span,
            "getter function appears to return the wrong field",
            "consider using",
            sugg,
            Applicability::MaybeIncorrect,
        );
    }
}
